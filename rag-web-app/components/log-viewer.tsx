"use client";

import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { FileText } from "lucide-react";

export function LogViewer() {
  const [open, setOpen] = useState(false);
  const [log, setLog] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchLog = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/logs/query");
      if (!res.ok) throw new Error("Inizia una nuova conversazione per visualizzare i log aggiornati.");
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
        <Button variant="outline" className="w-full gap-2" onClick={fetchLog}>
          <FileText className="h-4 w-4" />
          Visualizza Log Query
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Log delle Query</DialogTitle>
        </DialogHeader>
        {loading && <div className="text-sm">Caricamento...</div>}
        {error && <div className="text-sm text-destructive">{error}</div>}
        {log && (
          <div className="max-h-[60vh] overflow-auto bg-muted p-2 rounded">
            {log.split('\n').filter(Boolean).map((line, idx) => {
              let obj: any = null;
              try {
                obj = JSON.parse(line);
              } catch {
                return (
                  <pre key={idx} className="text-xs text-destructive">{line}</pre>
                );
              }
              return (
                <table key={idx} className="mb-4 w-full text-xs border border-border bg-background rounded">
                  <tbody>
                    {Object.entries(obj).map(([k, v]) => {
                      let displayValue = v;
                      // Format ISO timestamp fields
                      if (typeof v === 'string' && k.toLowerCase().includes('timestamp')) {
                        const date = new Date(v);
                        if (!isNaN(date.getTime())) {
                          const pad = (n: number) => n.toString().padStart(2, '0');
                          displayValue = `${pad(date.getDate())}/${pad(date.getMonth()+1)}/${date.getFullYear()} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
                        }
                      }
                      return (
                        <tr key={k} className="align-top border-b last:border-b-0 border-border">
                          <td className="font-semibold pr-2 py-1 text-nowrap text-muted-foreground align-top min-w-[120px]">{k}</td>
                          <td className="py-1 break-all">
                            {typeof v === 'object' && v !== null ? (
                              <pre className="whitespace-pre-wrap break-all bg-transparent p-0 m-0">{JSON.stringify(v, null, 2)}</pre>
                            ) : String(displayValue)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              );
            })}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
