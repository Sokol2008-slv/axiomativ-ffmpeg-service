import express from 'express'
import { processVideoJob } from './processor.js'

const app = express()
app.use(express.json({ limit: '10mb' }))

const PORT = process.env.PORT || 3001
const SECRET = process.env.FFMPEG_SERVICE_SECRET || 'dev-secret'

// Health check
app.get('/health', (req, res) => {
  res.json({ ok: true, service: 'axiomativ-ffmpeg', time: new Date().toISOString() })
})

// Главный эндпоинт — вызывается из Inngest
app.post('/process', async (req, res) => {
  // Проверяем секрет
  const authHeader = req.headers.authorization
  if (authHeader !== `Bearer ${SECRET}`) {
    return res.status(401).json({ error: 'Unauthorized' })
  }

  const { jobId, userId } = req.body
  if (!jobId || !userId) {
    return res.status(400).json({ error: 'jobId and userId required' })
  }

  // Отвечаем сразу (не ждём завершения)
  res.json({ ok: true, jobId, message: 'Processing started' })

  // Обрабатываем асинхронно
  processVideoJob(jobId, userId).catch(err => {
    console.error(`[Job ${jobId}] Fatal error:`, err)
  })
})

app.listen(PORT, () => {
  console.log(`FFmpeg service running on port ${PORT}`)
})
