import express from 'express'
import cors from 'cors'
import multer from 'multer'
import os from 'os'
import path from 'path'
import fs from 'fs'
import { processVideoJob } from './processor.js'
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
  processVideoJob(job.id, userId, req.file.path).catch(err => {
    console.error(`[Job ${job.id}] Fatal:`, err)
  })
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

app.listen(PORT, () => {
  console.log(`FFmpeg service running on port ${PORT}`)
})
