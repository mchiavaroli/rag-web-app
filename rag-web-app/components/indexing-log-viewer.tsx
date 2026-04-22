"use client";

import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { FileText } from "lucide-react";

export function IndexingLogViewer() {
  const [open, setOpen] = useState(false);
  const [log, setLog] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchLog = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/logs/indexing");
      if (!res.ok) throw new Error("Errore nel recupero del log");
      const text = await res.text();
      setLog(text);
    } catch (e: any) {
      setError(e.message || "Errore sconosciuto");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="w-full gap-2 mb-2" onClick={fetchLog}>
          <FileText className="h-4 w-4" />
          Visualizza Log Indicizzazione
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Log di Indicizzazione</DialogTitle>
        </DialogHeader>
        {loading && <div className="text-sm">Caricamento...</div>}
        {error && <div className="text-sm text-destructive">{error}</div>}
        {log && (() => {
          // Funzione per formattare la data
          function formatDate(val: string | number) {
            if (!val) return '';
            let d: Date;
            if (typeof val === 'number') {
              d = new Date(val * 1000);
            } else if (/^\d{4}-\d{2}-\d{2}T/.test(val)) {
              d = new Date(val);
            } else if (/^\d+$/.test(val)) {
              d = new Date(Number(val) * 1000);
            } else {
              d = new Date(val);
            }
            if (isNaN(d.getTime())) return String(val);
            const pad = (n: number) => n.toString().padStart(2, '0');
            return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
          }

          // Mostra solo l'ultima entry valida
          const lines = log.split('\n').filter(Boolean);
          let lastObj = null;
          for (let i = lines.length - 1; i >= 0; i--) {
            try {
              lastObj = JSON.parse(lines[i]);
              break;
            } catch {}
          }
          if (!lastObj) return <div className="text-xs text-destructive">Nessun log valido trovato.</div>;

          return (
            <div className="max-h-[60vh] overflow-auto bg-muted p-2 rounded">
              <table className="mb-4 w-full text-xs border border-border bg-background rounded">
                <tbody>
                  {Object.entries(lastObj).map(([k, v]) => (
                    <tr key={k} className="align-top border-b last:border-b-0 border-border">
                      <td className="font-semibold pr-2 py-1 text-nowrap text-muted-foreground align-top min-w-[120px]">{k}</td>
                      <td className="py-1 break-all">
                        {(() => {
                          const key = k.toLowerCase();
                          if (key === 'processing_time_ms') return String(v);
                          if ((key === 'timestamp' || key === 'started_at')) {
                            if (typeof v === 'string' && /^\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}/i.test(v)) {
                              return v.replace('T', ' ').replace('Z', '');
                            } else {
                              return formatDate(v);
                            }
                          }
                          if ((key.includes('time') || key.includes('date')) && key !== 'processing_time_ms') {
                            return formatDate(v);
                          }
                          if (typeof v === 'object' && v !== null) {
                            return <pre className="whitespace-pre-wrap break-all bg-transparent p-0 m-0">{JSON.stringify(v, null, 2)}</pre>;
                          }
                          return String(v);
                        })()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })()}
      </DialogContent>
    </Dialog>
  );
}
