import ffmpegStatic from 'ffmpeg-static'
import ffmpeg from 'fluent-ffmpeg'
import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import fs from 'fs'
import path from 'path'
import os from 'os'
import { randomUUID } from 'crypto'
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

    // 4. Авто-кадрирование в 9:16 (если горизонтальное видео)
    if (job.opt_reframe) {
      const reframedPath = path.join(tmpDir, 'reframed.mp4')
      console.log(`[${jobId}] Reframing to 9:16...`)
      await reframeToVertical(currentPath, reframedPath)
      currentPath = reframedPath
      console.log(`[${jobId}] Reframe done`)
    }

    // 5. Транскрипция через Whisper (если нужны субтитры или удаление пауз)
    let wordTimestamps = []
    let srtPath = null

    if (job.opt_fillers || job.opt_subtitles) {
      console.log(`[${jobId}] Transcribing with Whisper...`)
      const transcriptResult = await transcribeWithWhisper(currentPath, job.opt_subtitles_lang)
      wordTimestamps = transcriptResult.words || []
      console.log(`[${jobId}] Whisper got ${wordTimestamps.length} words`)
    }

    // 5. Авто-хук: найти самый цепляющий момент и поставить его в начало
    if (job.opt_hook && wordTimestamps.length > 0) {
      const hookResult = await findHookSegment(wordTimestamps)
      if (hookResult) {
        const hookedPath = path.join(tmpDir, 'hooked.mp4')
        await prependHook(currentPath, hookedPath, hookResult)
        currentPath = hookedPath
        // Пересчитываем таймстемпы: хук добавлен в начало
        const hookDuration = hookResult.end - hookResult.start
        wordTimestamps = [
          ...wordTimestamps.filter(w => w.start >= hookResult.start && w.end <= hookResult.end)
            .map(w => ({ ...w, start: w.start - hookResult.start, end: w.end - hookResult.start })),
          ...wordTimestamps.map(w => ({ ...w, start: w.start + hookDuration, end: w.end + hookDuration })),
        ]
        console.log(`[${jobId}] Hook prepended: ${hookResult.start.toFixed(2)}s–${hookResult.end.toFixed(2)}s`)
      }
    }

    // 6. Удаляем паузы и слова-паразиты
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

    // 11. Удаление файла происходит при следующем старте сервиса (cleanupExpiredFiles)
    // setTimeout не используем — Railway перезапускает контейнер и таймер теряется

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

// ── Режим нарезки на клипы ───────────────────────────────────────────────────
export async function processClipsJob(jobId, userId, uploadedFilePath = null) {
  const supabase = getSupabase()
  const tmpDir = uploadedFilePath
    ? path.dirname(uploadedFilePath)
    : path.join(os.tmpdir(), `job_${jobId}_${randomUUID()}`)

  if (!uploadedFilePath) fs.mkdirSync(tmpDir, { recursive: true })
  console.log(`[${jobId}] CLIPS MODE: starting in ${tmpDir}`)

  try {
    const { data: job } = await supabase.from('video_jobs').select('*').eq('id', jobId).eq('user_id', userId).single()
    if (!job) throw new Error(`Job not found: ${jobId}`)

    await supabase.from('video_jobs').update({ status: 'processing', started_at: new Date().toISOString() }).eq('id', jobId)

    // Берём файл
    let inputPath = uploadedFilePath && fs.existsSync(uploadedFilePath) ? uploadedFilePath : null
    if (!inputPath) {
      const { data: signed } = await supabase.storage.from('videos').createSignedUrl(job.storage_path, 3600)
      inputPath = path.join(tmpDir, `input_${Date.now()}.mp4`)
      await downloadFile(signed.signedUrl, inputPath)
    }

    // Нормализуем
    const normalizedPath = path.join(tmpDir, 'normalized.mp4')
    await normalizeVideo(inputPath, normalizedPath)
    const totalDuration = await getVideoDuration(normalizedPath)
    console.log(`[${jobId}] Duration: ${totalDuration.toFixed(0)}s`)

    // Транскрибируем с поддержкой длинных видео (чанками по 20 мин)
    const words = await transcribeWithChunking(normalizedPath, job.opt_subtitles_lang, totalDuration)
    console.log(`[${jobId}] Total words: ${words.length}`)

    // GPT находит лучшие сегменты
    const clipsCount = job.opt_clips_count || 5
    const segments = await findBestClipSegments(words, totalDuration, clipsCount)
    console.log(`[${jobId}] Found ${segments.length} clip segments`)

    // Обрабатываем каждый клип
    const EXPIRES_IN = 60 * 60 * 24
    const resultClips = []

    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i]
      const clipPath = path.join(tmpDir, `clip_${i}.mp4`)
      console.log(`[${jobId}] Processing clip ${i + 1}/${segments.length}: ${seg.start.toFixed(0)}s–${seg.end.toFixed(0)}s`)

      // Вырезаем сегмент
      await extractClip(normalizedPath, clipPath, seg.start, seg.end)

      // Авто-кадрирование в 9:16
      let processedClipPath = clipPath
      if (job.opt_reframe) {
        const reframedClipPath = path.join(tmpDir, `clip_${i}_reframed.mp4`)
        await reframeToVertical(clipPath, reframedClipPath)
        processedClipPath = reframedClipPath
      }

      // Слова для этого сегмента
      const clipWords = words
        .filter(w => w.start >= seg.start && w.end <= seg.end)
        .map(w => ({ ...w, start: w.start - seg.start, end: w.end - seg.start }))

      // Субтитры
      let assPath = null
      if (job.opt_subtitles && clipWords.length > 0) {
        const dims = await getVideoDimensions(processedClipPath)
        assPath = path.join(tmpDir, `clip_${i}.ass`)
        generateASS(clipWords, assPath, dims.width, dims.height)
      }

      // Цветокоррекция + субтитры
      const finalPath = path.join(tmpDir, `clip_${i}_final.mp4`)
      if (job.opt_subtitles || job.opt_color) {
        await applyFilters(processedClipPath, finalPath, { srtPath: assPath, color: job.opt_color })
      } else {
        fs.copyFileSync(processedClipPath, finalPath)
      }

      // Загружаем в Storage
      const storagePath = `${userId}/clip_${jobId}_${i}.mp4`
      const buf = fs.readFileSync(finalPath)
      await supabase.storage.from('videos').upload(storagePath, buf, { contentType: 'video/mp4', upsert: true })

      const { data: signed } = await supabase.storage.from('videos').createSignedUrl(storagePath, EXPIRES_IN)
      resultClips.push({
        index: i + 1,
        title: seg.title || `Клип ${i + 1}`,
        duration: Math.round(seg.end - seg.start),
        url: signed?.signedUrl || null,
        path: storagePath,
      })

      // Сообщаем о прогрессе
      await supabase.from('video_jobs').update({
        error_message: `Обрабатываю клип ${i + 1} из ${segments.length}...`
      }).eq('id', jobId)
    }

    const expiresAt = new Date(Date.now() + EXPIRES_IN * 1000).toISOString()
    await supabase.from('video_jobs').update({
      status: 'done',
      result_clips: resultClips,
      finished_at: new Date().toISOString(),
      expires_at: expiresAt,
      error_message: null,
    }).eq('id', jobId)

    // Удаляем оригинал
    await supabase.storage.from('videos').remove([job.storage_path])
    console.log(`[${jobId}] CLIPS DONE: ${resultClips.length} clips`)

  } catch (err) {
    console.error(`[${jobId}] Clips error:`, err)
    await supabase.from('video_jobs').update({
      status: 'error',
      error_message: err instanceof Error ? err.message : String(err),
      finished_at: new Date().toISOString(),
    }).eq('id', jobId)
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  }
}

// Транскрипция с поддержкой длинных видео (чанки по 20 минут)
async function transcribeWithChunking(videoPath, language, totalDuration) {
  const CHUNK_SEC = 20 * 60 // 20 минут

  if (totalDuration <= CHUNK_SEC) {
    const result = await transcribeWithWhisper(videoPath, language)
    return result.words || []
  }

  // Разбиваем на чанки
  const allWords = []
  const chunkCount = Math.ceil(totalDuration / CHUNK_SEC)

  for (let i = 0; i < chunkCount; i++) {
    const chunkStart = i * CHUNK_SEC
    const chunkEnd = Math.min((i + 1) * CHUNK_SEC, totalDuration)
    const chunkAudio = path.join(path.dirname(videoPath), `chunk_${i}.wav`)

    // Вырезаем аудио-чанк
    await new Promise((resolve, reject) => {
      ffmpeg(videoPath)
        .setStartTime(chunkStart)
        .duration(chunkEnd - chunkStart)
        .outputOptions(['-vn', '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', '-f', 'wav'])
        .output(chunkAudio)
        .on('end', resolve).on('error', reject).run()
    })

    try {
      const result = await transcribeWithWhisper(chunkAudio, language)
      const chunkWords = (result.words || []).map(w => ({
        ...w,
        start: w.start + chunkStart,
        end: w.end + chunkStart,
      }))
      allWords.push(...chunkWords)
      console.log(`Chunk ${i + 1}/${chunkCount}: ${chunkWords.length} words`)
    } finally {
      try { fs.unlinkSync(chunkAudio) } catch { /* ignore */ }
    }
  }

  return allWords
}

// GPT находит N лучших сегментов для Reels
async function findBestClipSegments(words, totalDuration, count) {
  if (words.length < 5) {
    // Fallback: нарезаем равномерно
    const segDur = Math.min(60, totalDuration / count)
    return Array.from({ length: count }, (_, i) => ({
      start: i * (totalDuration / count),
      end: i * (totalDuration / count) + segDur,
      title: `Клип ${i + 1}`,
    }))
  }

  const openai = getOpenAI()
  // Берём каждое 3-е слово чтобы не превысить лимит токенов
  const sample = words.filter((_, i) => i % 3 === 0)
  const transcript = sample.map(w => `[${w.start.toFixed(0)}s] ${w.word}`).join(' ')

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{
        role: 'user',
        content: `Ты — эксперт по вирусному контенту для Instagram Reels и TikTok.

Транскрипт видео (${Math.round(totalDuration / 60)} мин) с таймкодами:
${transcript}

Найди РОВНО ${count} лучших момента для отдельных Reels (каждый 30–90 секунд).
Критерии: законченная мысль, интересная история, полезный совет, смешной момент.
Сегменты НЕ должны пересекаться.

Ответь ТОЛЬКО JSON без markdown:
[{"start": <сек>, "end": <сек>, "title": "<название клипа>", "reason": "<почему>"}, ...]`,
      }],
      max_tokens: 500,
      temperature: 0.3,
    })

    const text = response.choices[0]?.message?.content?.trim() || '[]'
    const segments = JSON.parse(text)

    return segments
      .filter(s => typeof s.start === 'number' && typeof s.end === 'number' && s.end - s.start >= 15)
      .slice(0, count)
  } catch (e) {
    console.warn('findBestClipSegments failed:', e.message)
    // Fallback: равномерная нарезка
    const segDur = Math.min(60, totalDuration / count)
    return Array.from({ length: count }, (_, i) => ({
      start: i * (totalDuration / count),
      end: i * (totalDuration / count) + segDur,
      title: `Клип ${i + 1}`,
    }))
  }
}

// Вырезаем один клип
function extractClip(inputPath, outputPath, start, end) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .setStartTime(start)
      .duration(end - start)
      .outputOptions(['-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-b:a', '128k'])
      .output(outputPath)
      .on('end', resolve).on('error', reject).run()
  })
}

// ── Транскрипция через Whisper ──────────────────────────────────────────────
// Whisper API: максимум 25 МБ — сначала извлекаем аудио (AAC/m4a, без MP3!)
async function transcribeWithWhisper(videoPath, language) {
  const openai = getOpenAI()

  // WAV (PCM) — никаких кодеков, всегда работает. 37с моно 16кГц = ~1.2МБ
  const audioPath = path.join(path.dirname(videoPath), 'whisper_audio.wav')
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
    try { fs.unlinkSync(audioPath) } catch { /* ignore */ }
  }
}

// Нормализация видео: scale до 1080p, конвертация в H.264 MP4
function normalizeVideo(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions([
        // scale: ширина и высота не больше 1080, сохраняем пропорции
        '-vf', 'scale=w=\'if(gt(iw,ih),min(1920,iw),-2)\':h=\'if(gt(ih,iw),min(1920,ih),-2)\'',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-threads', '2',
      ])
      .output(outputPath)
      .on('end', resolve)
      .on('error', (err) => {
        // Fallback: простая конвертация без scale если vf упал
        ffmpeg(inputPath)
          .outputOptions(['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', '-threads', '2'])
          .output(outputPath)
          .on('end', resolve)
          .on('error', reject)
          .run()
      })
      .run()
  })
}

// Извлечение аудио в WAV PCM — без кодеков, гарантированно работает
// mono 16kHz 16-bit = ~1.2МБ/минуту, Whisper принимает WAV
function extractAudio(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions([
        '-vn',                // без видео
        '-ac', '1',           // mono
        '-ar', '16000',       // 16kHz
        '-acodec', 'pcm_s16le', // RAW PCM — нет кодека, нет проблем
        '-f', 'wav',          // WAV контейнер
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
    // Эканье — главный враг (е/э с длительностью >= 0.25с)
    'е', 'э',
    // Повторы
    'ее', 'еее', 'ееее',
    'ээ', 'эээ', 'эээ',
    'эм', 'эмм', 'эммм', 'эм-м',
    'ммм', 'мм', 'хм', 'хмм',
    'ааа', 'аааа',
    // Английские
    'um', 'uh', 'uhh', 'uhm', 'hmm', 'hm', 'erm',
    // ЗАЩИЩЁННЫЕ (никогда не резать):
    // 'є' (U+0454) — украинское "есть/являются" — НЕ в списке
    // 'а'           — союз "а" — НЕ в списке
  ])

  // Regex: 1+ гласных подряд (е, э, ее, эээ...) — НО не "є" (другой символ U+0454)
  const FILLER_PATTERN = /^[еэ]{1,}$/i

  const MIN_PAUSE = 0.45
  const KEEP_PAUSE = 0.15

  // Минимальная длительность: 0.4с — только ДЛИННЫЕ "э-э-э", короткие не трогаем
  const MIN_FILLER_DURATION = 0.40
  const MAX_FILLER_DURATION = 2.5

  const segments = []

  for (let i = 0; i < words.length; i++) {
    const w = words[i]
    const clean = w.word.toLowerCase().replace(/[.,!?…\-–]/g, '').trim()
    const duration = w.end - w.start

    const isFiller = FILLER_WORDS.has(clean) || FILLER_PATTERN.test(clean)
    if (isFiller && duration >= MIN_FILLER_DURATION && duration <= MAX_FILLER_DURATION) {
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
      // -42dB и 0.25с — ловим эканье и паузы, не режем слоги
      .outputOptions(['-af', 'silencedetect=noise=-42dB:duration=0.25', '-f', 'null'])
      .output('/dev/null')
      .on('stderr', (line) => {
        const startMatch = line.match(/silence_start: ([\d.]+)/)
        const endMatch = line.match(/silence_end: ([\d.]+)/)
        if (startMatch) currentStart = parseFloat(startMatch[1])
        if (endMatch && currentStart !== null) {
          const end = parseFloat(endMatch[1])
          const duration = end - currentStart
          // Паузы 0.25–2с
          if (duration >= 0.25 && duration <= 2.0) {
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
    const text = chunk.map(w => w.word.trim()).filter(Boolean).join(' ')
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

// ── Авто-хук: находим самый цепляющий момент через OpenAI ───────────────────
// Возвращает { start, end } сегмента который нужно поставить в начало
async function findHookSegment(words) {
  if (words.length < 10) return null

  const openai = getOpenAI()

  // Собираем транскрипт с таймкодами для GPT
  const transcript = words.map(w => `[${w.start.toFixed(1)}] ${w.word}`).join(' ')

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{
        role: 'user',
        content: `Ты — эксперт по созданию вирусного контента для Instagram Reels и TikTok.

Вот транскрипт видео с таймкодами (формат [секунда] слово):
${transcript}

Найди ОДИН момент (15–45 секунд) который:
- Начинается с интригующей фразы, вопроса или сильного утверждения
- Заставит зрителя хотеть досмотреть видео до конца
- Лучше всего подходит как "хук" для начала Reels/TikTok

Ответь ТОЛЬКО JSON без markdown:
{"start": <секунда начала>, "end": <секунда конца>, "reason": "<1 предложение почему>"}`,
      }],
      max_tokens: 100,
      temperature: 0.3,
    })

    const text = response.choices[0]?.message?.content?.trim() || ''
    const json = JSON.parse(text)

    if (typeof json.start === 'number' && typeof json.end === 'number') {
      const duration = json.end - json.start
      if (duration >= 5 && duration <= 60) {
        console.log(`Hook found: ${json.start}s–${json.end}s — ${json.reason}`)
        return { start: json.start, end: json.end }
      }
    }
  } catch (e) {
    console.warn('findHookSegment failed:', e.message)
  }
  return null
}

// Вставляем хук в начало видео через FFmpeg concat
function prependHook(inputPath, outputPath, hook) {
  return new Promise((resolve, reject) => {
    // Вырезаем хук-сегмент + добавляем оригинал после него
    const filterComplex = [
      `[0:v]trim=start=${hook.start}:end=${hook.end},setpts=PTS-STARTPTS[hv]`,
      `[0:a]atrim=start=${hook.start}:end=${hook.end},asetpts=PTS-STARTPTS[ha]`,
      `[0:v]setpts=PTS-STARTPTS[ov]`,
      `[0:a]asetpts=PTS-STARTPTS[oa]`,
      `[hv][ha][ov][oa]concat=n=2:v=1:a=1[outv][outa]`,
    ].join(';')

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

// ── Авто-кадрирование в 9:16 (вертикальный формат для Reels/TikTok) ──────────
// Если видео горизонтальное — размываем фон и вписываем поверх него
function reframeToVertical(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(inputPath, (err, meta) => {
      if (err) return reject(err)
      const vs = meta.streams.find(s => s.codec_type === 'video')
      const w = vs?.width || 1920
      const h = vs?.height || 1080

      // Уже вертикальное (portrait) — просто копируем
      if (h >= w) {
        fs.copyFileSync(inputPath, outputPath)
        return resolve()
      }

      // Горизонтальное → blur pad: размытый фон + видео по центру
      // Фон: масштабируем чтобы заполнить 1080x1920, обрезаем, размываем
      // Передний план: масштабируем чтобы вписаться в 1080x1920, накладываем по центру
      const filterComplex = [
        '[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[bg_raw]',
        '[bg_raw]gblur=sigma=25[bg_blur]',
        '[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg]',
        '[bg_blur][fg]overlay=(W-w)/2:(H-h)/2[out]',
      ].join(';')

      ffmpeg(inputPath)
        .outputOptions([
          '-filter_complex', filterComplex,
          '-map', '[out]',
          '-map', '0:a?',
          '-c:v', 'libx264',
          '-preset', 'fast',
          '-crf', '23',
          '-c:a', 'copy',
        ])
        .output(outputPath)
        .on('end', resolve)
        .on('error', reject)
        .run()
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
