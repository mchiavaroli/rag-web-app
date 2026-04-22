import { NextRequest, NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

export async function GET() {
  try {
    // Path robusto rispetto alla working directory
    const logPath = path.resolve(process.cwd(), '../rag-backend/output/logs/query_logs.jsonl');
    const log = await fs.readFile(logPath, 'utf-8');
    return new NextResponse(log, {
      status: 200,
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    });
  } catch (error: any) {
    console.error('Errore lettura log:', error?.message || error);
    return NextResponse.json({ error: 'Impossibile leggere il file di log.', detail: error?.message }, { status: 500 });
  }
}
