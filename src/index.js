import express from 'express'
import cors from 'cors'
import multer from 'multer'
import os from 'os'
import path from 'path'
import fs from 'fs'
import { processVideoJob, processClipsJob } from './processor.js'
import { createClient } from '@supabase/supabase-js'
import { randomUUID } from 'crypto'

const app = express()
const PORT = process.env.PORT || 3001
const SECRET = process.env.FFMPEG_SERVICE_SECRET || 'dev-secret'

// CORS — разрешаем запросы с Vercel
app.use(cors({
  origin: [
    'https://axiomativ.vercel.app',
    'http://localhost:3000',
    /\.vercel\.app$/,
  ],
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'x-job-options'],
}))

app.use(express.json({ limit: '10mb' }))

// Multer — принимаем файлы любого размера на диск
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      const tmpDir = path.join(os.tmpdir(), `upload_${randomUUID()}`)
      fs.mkdirSync(tmpDir, { recursive: true })
      req._uploadDir = tmpDir
      cb(null, tmpDir)
    },
    filename: (req, file, cb) => cb(null, file.originalname),
  }),
  limits: { fileSize: 3 * 1024 * 1024 * 1024 }, // 3 GB
})

function getSupabase() {
  return createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY)
}

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({ ok: true, service: 'axiomativ-ffmpeg', time: new Date().toISOString() })
})

// ── Прямая загрузка видео с браузера → обработка ─────────────────────────────
app.post('/upload', upload.single('file'), async (req, res) => {
  // Верифицируем Supabase JWT пользователя
  const authHeader = req.headers.authorization
  const token = authHeader?.replace('Bearer ', '')

  if (!token) {
    if (req.file) fs.rmSync(path.dirname(req.file.path), { recursive: true, force: true })
    return res.status(401).json({ error: 'No token provided' })
  }

  const supabase = getSupabase()
  const { data: { user }, error: authError } = await supabase.auth.getUser(token)

  if (authError || !user) {
    if (req.file) fs.rmSync(path.dirname(req.file.path), { recursive: true, force: true })
    return res.status(401).json({ error: 'Invalid or expired token' })
  }

  if (!req.file) {
    return res.status(400).json({ error: 'No file provided' })
  }

  // Опции из JSON-заголовка
  let options = {}
  try {
    options = JSON.parse(req.headers['x-job-options'] || '{}')
  } catch { /* ignore */ }

  const {
    optFillers = true,
    optSubtitles = true,
    optSubtitlesLang = 'auto',
    optColor = true,
    optHook = false,
    optClips = false,
    optClipsCount = 5,
  } = options

  const userId = user.id  // берём из верифицированного JWT

  // Создаём джоб в Supabase
  const { data: job, error: jobErr } = await supabase
    .from('video_jobs')
    .insert({
      user_id: userId,
      file_name: req.file.originalname,
      file_size: req.file.size,
      storage_path: `${userId}/${Date.now()}_${req.file.originalname}`,
      status: 'queued',
      opt_fillers: optFillers,
      opt_subtitles: optSubtitles,
      opt_subtitles_lang: optSubtitlesLang,
      opt_color: optColor,
      opt_hook: optHook,
      opt_clips: optClips,
      opt_clips_count: optClipsCount,
    })
    .select()
    .single()

  if (jobErr || !job) {
    fs.rmSync(path.dirname(req.file.path), { recursive: true, force: true })
    return res.status(500).json({ error: 'Failed to create job' })
  }

  // Отвечаем сразу — jobId готов, обработка идёт асинхронно
  res.json({ ok: true, jobId: job.id })

  // Обрабатываем асинхронно с уже скачанным файлом
  if (optClips) {
    processClipsJob(job.id, userId, req.file.path).catch(err => {
      console.error(`[Job ${job.id}] Fatal:`, err)
    })
  } else {
    processVideoJob(job.id, userId, req.file.path).catch(err => {
      console.error(`[Job ${job.id}] Fatal:`, err)
    })
  }
})

// ── Вызов из Inngest (если нужно) ─────────────────────────────────────────────
app.post('/process', async (req, res) => {
  const authHeader = req.headers.authorization
  if (authHeader !== `Bearer ${SECRET}`) {
    return res.status(401).json({ error: 'Unauthorized' })
  }

  const { jobId, userId } = req.body
  if (!jobId || !userId) {
    return res.status(400).json({ error: 'jobId and userId required' })
  }

  res.json({ ok: true, jobId })
  processVideoJob(jobId, userId, null).catch(err => {
    console.error(`[Job ${jobId}] Fatal:`, err)
  })
})

// ── Отмена джоба (статус queued → cancelled) ─────────────────────────────────
app.post('/cancel', async (req, res) => {
  const authHeader = req.headers.authorization
  const token = authHeader?.replace('Bearer ', '')
  if (!token) return res.status(401).json({ error: 'No token' })

  const supabase = getSupabase()
  const { data: { user }, error: authError } = await supabase.auth.getUser(token)
  if (authError || !user) return res.status(401).json({ error: 'Invalid token' })

  const { jobId } = req.body
  if (!jobId) return res.status(400).json({ error: 'jobId required' })

  const { error } = await supabase
    .from('video_jobs')
    .update({ status: 'error', error_message: 'Отменено пользователем', finished_at: new Date().toISOString() })
    .eq('id', jobId)
    .eq('user_id', user.id)
    .eq('status', 'queued') // только queued, нельзя отменить processing

  if (error) return res.status(500).json({ error: error.message })
  res.json({ ok: true })
})

app.listen(PORT, async () => {
  console.log(`FFmpeg service running on port ${PORT}`)

  // При старте — сбрасываем джобы застрявшие в processing/queued (Railway рестарт)
  try {
    const supabase = getSupabase()
    const { data: stuckJobs } = await supabase
      .from('video_jobs')
      .select('id')
      .in('status', ['processing', 'queued'])

    if (stuckJobs?.length) {
      await supabase
        .from('video_jobs')
        .update({
          status: 'error',
          error_message: 'Сервис перезапустился во время обработки. Попробуйте ещё раз.',
          finished_at: new Date().toISOString(),
        })
        .in('id', stuckJobs.map(j => j.id))
      console.log(`Recovered ${stuckJobs.length} stuck jobs on startup`)
    }

    // Удаляем файлы из Storage у джобов с истёкшим expires_at
    const { data: expiredJobs } = await supabase
      .from('video_jobs')
      .select('id, result_path')
      .eq('status', 'done')
      .lt('expires_at', new Date().toISOString())
      .not('result_path', 'is', null)

    if (expiredJobs?.length) {
      const paths = expiredJobs.map(j => j.result_path).filter(Boolean)
      await supabase.storage.from('videos').remove(paths)
      await supabase.from('video_jobs')
        .update({ result_path: null, result_url: null })
        .in('id', expiredJobs.map(j => j.id))
      console.log(`Cleaned up ${expiredJobs.length} expired files on startup`)
    }
  } catch (e) {
    console.warn('Startup recovery failed:', e.message)
  }
})
