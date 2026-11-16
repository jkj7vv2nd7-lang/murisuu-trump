#!/usr/bin/env python3
"""Loopback-only web bridge between the AI Teacher page and local Ollama."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from pathlib import Path
import json

HOST = "127.0.0.1"
PORT = 8765
MODEL = "qwen3.5:4b"
OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"
ROOT = Path(__file__).resolve().parent

SYSTEM_PROMPT = """あなたは日本の中学生向けの、親切で正確な数学の先生です。
必ず日本語で、指定された学年に合う言葉を使って説明してください。
最初に結論を短く示し、その後に考え方、途中式、確認方法の順で説明してください。
一度に多くのことを詰め込まず、数学記号は読みやすく書いてください。
問題文が不明確なら、勝手に条件を補わず確認してください。
誤りを指摘されたら、言い訳せずに計算を見直してください。
生徒を責めたり、からかったりせず、落ち着いた励ましの言葉を使ってください。"""

class Handler(BaseHTTPRequestHandler):
    def _send(self, status, data, content_type="application/json; charset=utf-8"):
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            try:
                req = Request("http://127.0.0.1:11434/api/tags")
                with urlopen(req, timeout=5) as response:
                    tags = json.loads(response.read().decode("utf-8"))
                names = [m.get("name", "") for m in tags.get("models", [])]
                if not any(n == MODEL or n.startswith(MODEL + ":") for n in names):
                    self._send(503, {"error": f"Ollamaは起動していますが、モデル {MODEL} が見つかりません。"})
                    return
                self._send(200, {"ok": True, "model": MODEL})
            except Exception as exc:
                self._send(503, {"error": f"Ollamaに接続できません: {exc}"})
            return
        if self.path in ("/", "/index.html"):
            try:
                self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            except OSError:
                self._send(500, {"error": "index.html が見つかりません。"})
            return
        self._send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            self._send(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 12000:
                self._send(400, {"error": "質問の長さが不正です。"})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(payload.get("question", "")).strip()
            level = str(payload.get("level", "中学3年"))
            if not question or len(question) > 1200:
                self._send(400, {"error": "質問は1〜1200文字で入力してください。"})
                return
            if level not in ("中学1年", "中学2年", "中学3年", "高校入試の応用"):
                level = "中学3年"
            request_data = {
                "model": MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"学習段階：{level}\n質問：{question}"}
                ],
                "options": {"temperature": 0.3}
            }
            req = Request(OLLAMA_CHAT, data=json.dumps(request_data).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer = result.get("message", {}).get("content", "").strip()
            if not answer:
                self._send(502, {"error": "Ollamaから回答が返りませんでした。"})
                return
            self._send(200, {"answer": answer, "model": MODEL})
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            self._send(502, {"error": f"Ollama APIエラー ({exc.code}): {detail}"})
        except URLError as exc:
            self._send(503, {"error": f"Ollamaに接続できません: {exc.reason}"})
        except json.JSONDecodeError:
            self._send(400, {"error": "送信データを読み取れませんでした。"})
        except Exception as exc:
            self._send(500, {"error": f"サーバー内部エラー: {exc}"})

    def log_message(self, fmt, *args):
        print("[AI Teacher] " + (fmt % args))

if __name__ == "__main__":
    print(f"AI先生を起動しました: http://{HOST}:{PORT}")
    print("このサーバーはこのPC内 (127.0.0.1) からのみ接続できます。")
    print("終了するには Ctrl+C を押してください。")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
