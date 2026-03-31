import ffmpegStatic from 'ffmpeg-static'
import ffmpeg from 'fluent-ffmpeg'
import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import fs from 'fs'
import path from 'path'
import os from 'os'
import { randomUUID } from 'crypto'
import { Readable } from 'stream'

// Используем статичный ffmpeg бинарник
ffmpeg.setFfmpegPath(ffmpegStatic)

function getSupabase() {
  return createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_KEY
  )
}

function getOpenAI() {
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
}

// ── Главная функция обработки ──────────────────────────────────────────────
// uploadedFilePath — если файл уже на диске (прямая загрузка), null — скачиваем из Storage
export async function processVideoJob(jobId, userId, uploadedFilePath = null) {
  const supabase = getSupabase()
  const tmpDir = uploadedFilePath
    ? path.dirname(uploadedFilePath)
    : path.join(os.tmpdir(), `job_${jobId}_${randomUUID()}`)

  if (!uploadedFilePath) fs.mkdirSync(tmpDir, { recursive: true })

  console.log(`[${jobId}] Starting processing in ${tmpDir}`)

  try {
    // 1. Загружаем данные джоба
    const { data: job, error: jobErr } = await supabase
      .from('video_jobs')
      .select('*')
      .eq('id', jobId)
      .eq('user_id', userId)
      .single()

    if (jobErr || !job) throw new Error(`Job not found: ${jobId}`)

    // 2. Статус → processing
    await supabase.from('video_jobs').update({
      status: 'processing',
      started_at: new Date().toISOString()
    }).eq('id', jobId)

    // 3. Берём файл с диска или скачиваем из Storage
    let inputPath
    if (uploadedFilePath && fs.existsSync(uploadedFilePath)) {
      inputPath = uploadedFilePath
      console.log(`[${jobId}] Using uploaded file: ${inputPath}`)
    } else {
      const { data: signedData } = await supabase.storage
        .from('videos')
        .createSignedUrl(job.storage_path, 3600)
      if (!signedData?.signedUrl) throw new Error('Failed to get signed URL')
      inputPath = path.join(tmpDir, `input_${Date.now()}.mp4`)
      await downloadFile(signedData.signedUrl, inputPath)
      console.log(`[${jobId}] Downloaded: ${inputPath}`)
    }

    let currentPath = inputPath

    // 4. Транскрипция через Whisper (если нужны субтитры или удаление пауз)
    let wordTimestamps = []
    let srtPath = null

    if (job.opt_fillers || job.opt_subtitles) {
      console.log(`[${jobId}] Transcribing with Whisper...`)
      const transcriptResult = await transcribeWithWhisper(currentPath, job.opt_subtitles_lang)
      wordTimestamps = transcriptResult.words || []

      // Генерируем SRT если нужны субтитры
      if (job.opt_subtitles && wordTimestamps.length > 0) {
        srtPath = path.join(tmpDir, 'subtitles.srt')
        generateSRT(wordTimestamps, srtPath)
        console.log(`[${jobId}] SRT generated: ${srtPath}`)
      }
    }

    // 5. Удаляем паузы и слова-паразиты
    if (job.opt_fillers && wordTimestamps.length > 0) {
      const outputPath = path.join(tmpDir, `no_fillers.mp4`)
      const cutSegments = detectFillerSegments(wordTimestamps)
      console.log(`[${jobId}] Cutting ${cutSegments.length} filler segments...`)
      await cutFillers(currentPath, outputPath, cutSegments)
      currentPath = outputPath
    }

    // 6. Субтитры + цветокоррекция (один проход FFmpeg)
    if (job.opt_subtitles || job.opt_color) {
      const outputPath = path.join(tmpDir, `processed.mp4`)
      console.log(`[${jobId}] Applying: subs=${job.opt_subtitles}, color=${job.opt_color}`)
      await applyFilters(currentPath, outputPath, {
        srtPath: job.opt_subtitles ? srtPath : null,
        color: job.opt_color,
      })
      currentPath = outputPath
    }

    // 7. Загружаем результат в Supabase Storage
    const resultStoragePath = job.storage_path.replace(/^([^/]+\/[^/]+)/, '$1_processed')
    const fileBuffer = fs.readFileSync(currentPath)

    const { error: uploadErr } = await supabase.storage
      .from('videos')
      .upload(resultStoragePath, fileBuffer, {
        contentType: 'video/mp4',
        upsert: true,
      })

    if (uploadErr) throw new Error(`Upload failed: ${uploadErr.message}`)
    console.log(`[${jobId}] Uploaded result: ${resultStoragePath}`)

    // 8. Удаляем оригинальный файл — он больше не нужен
    await supabase.storage.from('videos').remove([job.storage_path])
    console.log(`[${jobId}] Deleted original: ${job.storage_path}`)

    // 9. Signed URL на 24 часа (достаточно чтобы скачать)
    const EXPIRES_IN = 60 * 60 * 24  // 24 часа
    const { data: resultSigned } = await supabase.storage
      .from('videos')
      .createSignedUrl(resultStoragePath, EXPIRES_IN)

    const resultUrl = resultSigned?.signedUrl || null
    const expiresAt = new Date(Date.now() + EXPIRES_IN * 1000).toISOString()

    // 10. Статус → done
    await supabase.from('video_jobs').update({
      status: 'done',
      result_url: resultUrl,
      result_path: resultStoragePath,
      finished_at: new Date().toISOString(),
      expires_at: expiresAt,
    }).eq('id', jobId)

    console.log(`[${jobId}] Done! Expires at: ${expiresAt}`)

    // 11. Планируем удаление обработанного файла через 24 часа
    // (неблокирующий setTimeout — Railway держит процесс живым)
    setTimeout(async () => {
      try {
        await supabase.storage.from('videos').remove([resultStoragePath])
        console.log(`[${jobId}] Auto-deleted processed file after 24h`)
      } catch (e) {
        console.warn(`[${jobId}] Failed to auto-delete processed file:`, e)
      }
    }, EXPIRES_IN * 1000)

  } catch (err) {
    console.error(`[${jobId}] Error:`, err)
    try {
      await supabase.from('video_jobs').update({
        status: 'error',
        error_message: err instanceof Error ? err.message : String(err),
        finished_at: new Date().toISOString(),
      }).eq('id', jobId)
    } catch { /* ignore secondary error */ }
  } finally {
    // Чистим временные файлы
    fs.rmSync(tmpDir, { recursive: true, force: true })
    console.log(`[${jobId}] Cleaned up ${tmpDir}`)
  }
}

// ── Транскрипция через Whisper ──────────────────────────────────────────────
// Whisper API принимает максимум 25 МБ — сначала извлекаем аудио через FFmpeg
async function transcribeWithWhisper(videoPath, language) {
  const openai = getOpenAI()

  // Извлекаем аудио в mp3 (mono, 64k) — обычно < 10 МБ для видео до часа
  const audioPath = videoPath.replace(/\.[^/.]+$/, '') + '_audio.mp3'
  await extractAudio(videoPath, audioPath)

  try {
    const response = await openai.audio.transcriptions.create({
      file: fs.createReadStream(audioPath),
      model: 'whisper-1',
      response_format: 'verbose_json',
      timestamp_granularities: ['word'],
      language: language === 'auto' ? undefined : language,
    })
    return response
  } finally {
    // Удаляем временный аудиофайл
    try { fs.unlinkSync(audioPath) } catch { /* ignore */ }
  }
}

// Извлечение аудио в mp3 mono 64k
function extractAudio(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions([
        '-vn',          // без видео
        '-ac', '1',     // mono
        '-ar', '16000', // 16kHz — оптимально для Whisper
        '-b:a', '64k',  // 64 kbps
      ])
      .output(outputPath)
      .on('end', resolve)
      .on('error', reject)
      .run()
  })
}

// ── Детект слов-паразитов и пауз ────────────────────────────────────────────
function detectFillerSegments(words) {
  const FILLER_WORDS = new Set([
    // Русские
    'эм', 'эмм', 'эммм', 'ммм', 'мм', 'ааа', 'аааа', 'ну',
    'вот', 'типа', 'короче', 'блин', 'собственно', 'значит',
    // Английские
    'um', 'uh', 'uhh', 'uhm', 'hmm', 'hm', 'erm',
  ])
  const MIN_PAUSE = 0.65   // паузы длиннее этого вырезаем
  const KEEP_PAUSE = 0.20  // оставляем такую паузу вместо длинной

  const segments = []

  for (let i = 0; i < words.length; i++) {
    const w = words[i]
    const clean = w.word.toLowerCase().replace(/[.,!?…]/g, '').trim()

    // Слово-паразит
    if (FILLER_WORDS.has(clean)) {
      segments.push({ start: w.start, end: w.end })
    }

    // Пауза между словами
    if (i < words.length - 1) {
      const gap = words[i + 1].start - w.end
      if (gap > MIN_PAUSE) {
        segments.push({ start: w.end + KEEP_PAUSE, end: words[i + 1].start })
      }
    }
  }

  return mergeSegments(segments)
}

// Объединяем пересекающиеся сегменты
function mergeSegments(segs) {
  if (!segs.length) return []
  const sorted = [...segs].sort((a, b) => a.start - b.start)
  const merged = [sorted[0]]
  for (let i = 1; i < sorted.length; i++) {
    const last = merged[merged.length - 1]
    if (sorted[i].start <= last.end + 0.05) {
      last.end = Math.max(last.end, sorted[i].end)
    } else {
      merged.push(sorted[i])
    }
  }
  return merged
}

// ── Вырезаем паузы/паразиты через FFmpeg ────────────────────────────────────
async function cutFillers(inputPath, outputPath, cutSegments) {
  // Строим список "хороших" сегментов (всё что НЕ в cutSegments)
  // Получаем длительность видео
  const duration = await getVideoDuration(inputPath)
  const keepSegments = invertSegments(cutSegments, duration)

  if (keepSegments.length === 0) {
    fs.copyFileSync(inputPath, outputPath)
    return
  }

  // Строим filter_complex для concat
  // [0:v]trim=start=X:end=Y,setpts=PTS-STARTPTS[vN]
  // [0:a]atrim=start=X:end=Y,asetpts=PTS-STARTPTS[aN]
  // [v0][a0][v1][a1]...concat=n=N:v=1:a=1[outv][outa]
  const vParts = keepSegments.map((s, i) =>
    `[0:v]trim=start=${s.start}:end=${s.end},setpts=PTS-STARTPTS[v${i}]`
  )
  const aParts = keepSegments.map((s, i) =>
    `[0:a]atrim=start=${s.start}:end=${s.end},asetpts=PTS-STARTPTS[a${i}]`
  )
  const inputs = keepSegments.map((_, i) => `[v${i}][a${i}]`).join('')
  const concatFilter = `${inputs}concat=n=${keepSegments.length}:v=1:a=1[outv][outa]`
  const filterComplex = [...vParts, ...aParts, concatFilter].join(';')

  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions([
        '-filter_complex', filterComplex,
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
      ])
      .output(outputPath)
      .on('end', resolve)
      .on('error', reject)
      .run()
  })
}

// Получаем сегменты которые НАДО оставить (инверсия вырезаемых)
function invertSegments(cutSegs, totalDuration) {
  const keep = []
  let pos = 0

  for (const seg of cutSegs) {
    if (seg.start > pos + 0.05) {
      keep.push({ start: pos, end: seg.start })
    }
    pos = seg.end
  }

  if (pos < totalDuration - 0.05) {
    keep.push({ start: pos, end: totalDuration })
  }

  return keep
}

// ── Применяем субтитры и цветокоррекцию ─────────────────────────────────────
async function applyFilters(inputPath, outputPath, options) {
  const { srtPath, color } = options

  const vFilters = []

  // Цветокоррекция
  if (color) {
    // Насыщенность +30%, контраст +5%, небольшое осветление, чуть тепло
    vFilters.push('eq=saturation=1.30:contrast=1.05:brightness=0.02:gamma=0.97')
    // Лёгкое повышение резкости
    vFilters.push('unsharp=3:3:0.4')
    // Тёплый тон (чуть поднимаем красный, убираем синий)
    vFilters.push('colorbalance=rs=0.03:gs=0.0:bs=-0.03')
  }

  // Субтитры — должны идти ПОСЛЕДНИМИ (после цветокоррекции)
  if (srtPath) {
    // Экранируем путь для FFmpeg
    const escapedSrt = srtPath.replace(/\\/g, '/').replace(/:/g, '\\:')
    vFilters.push(
      `subtitles='${escapedSrt}':force_style='FontName=Arial,FontSize=22,Bold=1,Outline=2,Shadow=0,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,Alignment=2,MarginV=30'`
    )
  }

  const filterStr = vFilters.join(',')

  return new Promise((resolve, reject) => {
    const cmd = ffmpeg(inputPath)
      .outputOptions([
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '22',
        '-c:a', 'copy',
      ])

    if (filterStr) {
      cmd.videoFilter(filterStr)
    }

    cmd
      .output(outputPath)
      .on('end', resolve)
      .on('error', reject)
      .run()
  })
}

// ── Генерация SRT файла ──────────────────────────────────────────────────────
function generateSRT(words, outputPath) {
  const WORDS_PER_LINE = 7
  const lines = []
  let idx = 1

  for (let i = 0; i < words.length; i += WORDS_PER_LINE) {
    const chunk = words.slice(i, i + WORDS_PER_LINE)
    if (!chunk.length) continue

    const startSec = chunk[0].start
    const endSec = chunk[chunk.length - 1].end
    const text = chunk.map(w => w.word).join(' ').trim()

    if (!text) continue

    lines.push(
      `${idx}\n${srtTime(startSec)} --> ${srtTime(endSec)}\n${text}\n`
    )
    idx++
  }

  fs.writeFileSync(outputPath, lines.join('\n'), 'utf8')
}

function srtTime(sec) {
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = Math.floor(sec % 60)
  const ms = Math.round((sec % 1) * 1000)
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`
}

function pad(n, len = 2) {
  return String(n).padStart(len, '0')
}

// ── Получаем длительность видео ─────────────────────────────────────────────
function getVideoDuration(videoPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err, meta) => {
      if (err) reject(err)
      else resolve(meta.format.duration || 0)
    })
  })
}

// ── Скачиваем файл по URL ────────────────────────────────────────────────────
async function downloadFile(url, destPath) {
  const { default: fetch } = await import('node-fetch')
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Download failed: ${res.status}`)

  const arrayBuffer = await res.arrayBuffer()
  fs.writeFileSync(destPath, Buffer.from(arrayBuffer))
}
