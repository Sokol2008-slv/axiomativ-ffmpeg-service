import ffmpegStatic from 'ffmpeg-static'
import ffmpeg from 'fluent-ffmpeg'
import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import fs from 'fs'
import path from 'path'
import os from 'os'
import { randomUUID } from 'crypto'
import { Readable } from 'stream'
import { execSync } from 'child_process'

// Используем системный ffmpeg (nixpkgs) — он собран с libass/subtitles.
// Если недоступен — fallback на статичный бинарник.
let ffmpegPath = ffmpegStatic
try {
  const sys = execSync('which ffmpeg 2>/dev/null').toString().trim()
  if (sys) {
    ffmpegPath = sys
    console.log(`Using system ffmpeg: ${sys}`)
  }
} catch { /* ignore */ }
ffmpeg.setFfmpegPath(ffmpegPath)

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

    // 3.5. Конвертируем в 1080p MP4 — снижаем нагрузку на FFmpeg при обработке
    const normalizedPath = path.join(tmpDir, 'normalized.mp4')
    console.log(`[${jobId}] Normalizing to 1080p MP4...`)
    await normalizeVideo(inputPath, normalizedPath)
    let currentPath = normalizedPath

    // 4. Транскрипция через Whisper (если нужны субтитры или удаление пауз)
    let wordTimestamps = []
    let srtPath = null

    if (job.opt_fillers || job.opt_subtitles) {
      console.log(`[${jobId}] Transcribing with Whisper...`)
      const transcriptResult = await transcribeWithWhisper(currentPath, job.opt_subtitles_lang)
      wordTimestamps = transcriptResult.words || []
      console.log(`[${jobId}] Whisper got ${wordTimestamps.length} words`)
    }

    // 5. Удаляем паузы и слова-паразиты
    let cutSegments = []
    if (job.opt_fillers) {
      const outputPath = path.join(tmpDir, `no_fillers.mp4`)
      // Детект по Whisper timestamps
      const whisperCuts = wordTimestamps.length > 0 ? detectFillerSegments(wordTimestamps) : []
      // Детект тишины через FFmpeg (ловит "э" которые Whisper пропустил)
      const silenceCuts = await detectSilenceSegments(currentPath)
      // Объединяем оба метода
      cutSegments = mergeSegments([...whisperCuts, ...silenceCuts])
      console.log(`[${jobId}] Cutting ${cutSegments.length} segments (${whisperCuts.length} whisper + ${silenceCuts.length} silence)`)
      if (cutSegments.length > 0) {
        await cutFillers(currentPath, outputPath, cutSegments)
        currentPath = outputPath
      }
    }

    // 6. Генерируем ASS ПОСЛЕ вырезания — чтобы таймстемпы совпадали с новым видео
    if (job.opt_subtitles && wordTimestamps.length > 0) {
      const adjustedWords = adjustWordTimestamps(wordTimestamps, cutSegments)
      // Получаем размеры видео для точного масштабирования шрифта
      const dims = await getVideoDimensions(currentPath)
      srtPath = path.join(tmpDir, 'subtitles.ass')
      generateASS(adjustedWords, srtPath, dims.width, dims.height)
      console.log(`[${jobId}] ASS generated: ${srtPath} (${adjustedWords.length} words, ${dims.width}x${dims.height})`)
    }

    // 7. Субтитры + цветокоррекция (один проход FFmpeg)
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

// Нормализация видео: scale до 1080p, конвертация в H.264 MP4
// Снижает нагрузку на FFmpeg при дальнейшей обработке
function normalizeVideo(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions([
        '-vf', 'scale=\'min(1920,iw)\':\'min(1080,ih)\':force_original_aspect_ratio=decrease',
        '-c:v', 'libx264',
        '-preset', 'ultrafast', // быстрее = меньше RAM
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-threads', '2',
      ])
      .output(outputPath)
      .on('end', resolve)
      .on('error', reject)
      .run()
  })
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
    // Короткие звуки-паразиты
    'э', 'е', 'а',
    'эм', 'эмм', 'эммм', 'эм-м', 'ммм', 'мм', 'м',
    'ааа', 'аа', 'ааааа', 'эх',
    // Английские
    'um', 'uh', 'uhh', 'uhm', 'hmm', 'hm', 'erm', 'err',
  ])

  // Паттерны для regex-детекта: повторяющиеся гласные/согласные = звук-паразит
  const FILLER_PATTERN = /^[эеаиуоыёэ]{1,2}$/i  // одна-две гласных = "э", "е", "а"

  const MIN_PAUSE = 0.45   // паузы длиннее 0.45с вырезаем (было 0.65)
  const KEEP_PAUSE = 0.15  // оставляем небольшую паузу для естественности

  // Максимальная длительность слова-паразита — защита от случайных совпадений
  const MAX_FILLER_DURATION = 1.5

  const segments = []

  for (let i = 0; i < words.length; i++) {
    const w = words[i]
    const clean = w.word.toLowerCase().replace(/[.,!?…\-–]/g, '').trim()
    const duration = w.end - w.start

    // Слово-паразит: по списку или по паттерну коротких звуков
    const isFiller = FILLER_WORDS.has(clean) || FILLER_PATTERN.test(clean)
    if (isFiller && duration <= MAX_FILLER_DURATION) {
      // Добавляем небольшой отступ чтобы не обрезать соседние слова
      segments.push({
        start: Math.max(0, w.start - 0.02),
        end: w.end + 0.05,
      })
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

// ── Вырезаем паузы/паразиты через FFmpeg (с audio fade для плавности) ───────
async function cutFillers(inputPath, outputPath, cutSegments) {
  const duration = await getVideoDuration(inputPath)
  const keepSegments = invertSegments(cutSegments, duration)

  if (keepSegments.length === 0) {
    fs.copyFileSync(inputPath, outputPath)
    return
  }

  const FADE = 0.035 // 35мс аудио-фейд на каждом разрезе — убирает щелчки

  const vParts = keepSegments.map((s, i) =>
    `[0:v]trim=start=${s.start}:end=${s.end},setpts=PTS-STARTPTS[v${i}]`
  )
  const aParts = keepSegments.map((s, i) => {
    const segDur = s.end - s.start
    const fadeIn  = i > 0 ? `,afade=t=in:ss=0:d=${FADE}` : ''
    const fadeOut = i < keepSegments.length - 1
      ? `,afade=t=out:st=${Math.max(0, segDur - FADE)}:d=${FADE}`
      : ''
    return `[0:a]atrim=start=${s.start}:end=${s.end},asetpts=PTS-STARTPTS${fadeIn}${fadeOut}[a${i}]`
  })
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

  // Цветокоррекция — мягкая, как в Instagram/Reels (не перекручивать)
  if (color) {
    // Чуть поднимаем насыщенность и контраст — едва заметно
    vFilters.push('eq=saturation=1.10:contrast=1.02:brightness=0.01:gamma=1.0')
    // Минимальная резкость
    vFilters.push('unsharp=3:3:0.2')
  }

  // Субтитры — используем ASS фильтр (точный контроль размера под разрешение)
  if (srtPath) {
    const escapedAss = srtPath.replace(/\\/g, '/').replace(/:/g, '\\:')
    vFilters.push(`ass='${escapedAss}'`)
  }

  const filterStr = vFilters.join(',')

  return new Promise((resolve, reject) => {
    const cmd = ffmpeg(inputPath)
      .outputOptions([
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-level:v', '4.0',
        '-preset', 'fast',
        '-crf', '22',
        '-pix_fmt', 'yuv420p',   // максимальная совместимость
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-threads', '2',
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

// ── Детект тишины через FFmpeg (для "э" которые Whisper пропустил) ──────────
function detectSilenceSegments(videoPath) {
  return new Promise((resolve) => {
    const silences = []
    let currentStart = null

    ffmpeg(videoPath)
      .outputOptions(['-af', 'silencedetect=noise=-38dB:duration=0.2', '-f', 'null'])
      .output('/dev/null')
      .on('stderr', (line) => {
        const startMatch = line.match(/silence_start: ([\d.]+)/)
        const endMatch = line.match(/silence_end: ([\d.]+)/)
        if (startMatch) currentStart = parseFloat(startMatch[1])
        if (endMatch && currentStart !== null) {
          const end = parseFloat(endMatch[1])
          const duration = end - currentStart
          // Только короткие тишины (0.2–0.8с) — паузы-паразиты, не естественные паузы
          if (duration >= 0.2 && duration <= 0.8) {
            silences.push({ start: currentStart, end })
          }
          currentStart = null
        }
      })
      .on('end', () => resolve(silences))
      .on('error', () => resolve([])) // если ошибка — просто пустой список
      .run()
  })
}

// ── Корректировка таймстемпов слов после вырезания сегментов ─────────────────
// КРИТИЧНО: без этого субтитры будут смещены после фильтрации
function adjustWordTimestamps(words, cutSegments) {
  if (!cutSegments.length) return words

  return words
    .filter(w => {
      // Убираем слова которые полностью попали в вырезанный сегмент
      return !cutSegments.some(c => w.start >= c.start && w.end <= c.end + 0.1)
    })
    .map(w => {
      // Считаем сколько времени вырезано ДО этого слова
      let cutBefore = 0
      for (const c of cutSegments) {
        if (c.end <= w.start) {
          cutBefore += c.end - c.start
        } else if (c.start < w.start) {
          cutBefore += w.start - c.start
        }
      }
      return {
        ...w,
        start: Math.max(0, w.start - cutBefore),
        end: Math.max(0, w.end - cutBefore),
      }
    })
}

// ── Размеры видео ────────────────────────────────────────────────────────────
function getVideoDimensions(videoPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err, meta) => {
      if (err) return reject(err)
      const vs = meta.streams.find(s => s.codec_type === 'video')
      resolve({ width: vs?.width || 1080, height: vs?.height || 1920 })
    })
  })
}

// ── Генерация ASS субтитров (Instagram/Reels стиль) ──────────────────────────
// ASS формат позволяет точно задать FontSize относительно реального разрешения
function generateASS(words, outputPath, videoWidth = 1080, videoHeight = 1920) {
  const WORDS_PER_LINE = 3

  // FontSize: ~5.5% высоты видео → красиво для любого разрешения
  const fontSize = Math.round(videoHeight * 0.055)
  // Отступ снизу: ~7% высоты
  const marginV = Math.round(videoHeight * 0.07)

  // ASS цвета: &HAABBGGRR (alpha, blue, green, red)
  // Белый текст, чёрный контур, без фона
  const header = `[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: ${videoWidth}
PlayResY: ${videoHeight}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Liberation Sans,${fontSize},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0.3,0,1,2.5,1.0,2,40,40,${marginV},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text`

  const clean = words.filter(w => w.word && w.word.trim().length > 0)
  const dialogues = []

  for (let i = 0; i < clean.length; i += WORDS_PER_LINE) {
    const chunk = clean.slice(i, i + WORDS_PER_LINE)
    if (!chunk.length) continue
    const text = chunk.map(w => w.word).join('').replace(/\s+/g, ' ').trim()
    if (!text) continue
    const startT = assTime(chunk[0].start)
    const endT   = assTime(chunk[chunk.length - 1].end)
    dialogues.push(`Dialogue: 0,${startT},${endT},Default,,0,0,0,,${text}`)
  }

  fs.writeFileSync(outputPath, header + '\n' + dialogues.join('\n'), 'utf8')
}

function assTime(sec) {
  const h  = Math.floor(sec / 3600)
  const m  = Math.floor((sec % 3600) / 60)
  const s  = Math.floor(sec % 60)
  const cs = Math.round((sec % 1) * 100)
  return `${h}:${pad(m)}:${pad(s)}.${pad(cs)}`
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
