#!/usr/bin/env ruby
# Statischer Server + ML-Proxy für das Live-Audio-Analyse-Tool.
# Nur Ruby-Stdlib (socket, net/http) – keine Gems nötig.
#
#   ruby live-audio-tool/server.rb [PORT]
#
# Liefert die Web-App aus und leitet /ml/* an den Diarisierungs-Server
# weiter (Standard: http://127.0.0.1:8001, per ML_BACKEND übersteuerbar).
# Dadurch reicht eine einzige Adresse/ein einziger Tunnel – auch vom Handy.

require "socket"
require "net/http"
require "uri"

PORT = (ARGV[0] || 8000).to_i
BIND = ENV["BIND"] || "0.0.0.0"
ROOT = File.expand_path(__dir__)
ML_BACKEND = ENV["ML_BACKEND"] || "http://127.0.0.1:8001"

MIME = {
  ".html" => "text/html; charset=utf-8",
  ".js"   => "text/javascript; charset=utf-8",
  ".css"  => "text/css; charset=utf-8",
  ".json" => "application/json",
  ".svg"  => "image/svg+xml",
  ".png"  => "image/png",
  ".ico"  => "image/x-icon",
}.freeze

MAX_BODY = 200 * 1024 * 1024 # 200 MB – lange Aufnahmen sind groß

def respond(client, status, headers, body = "")
  client.write "HTTP/1.1 #{status}\r\n"
  headers.merge("Content-Length" => body.bytesize, "Connection" => "close").each do |k, v|
    client.write "#{k}: #{v}\r\n"
  end
  client.write "\r\n"
  client.write body
end

def read_headers(client)
  headers = {}
  while (line = client.gets) && line != "\r\n"
    key, value = line.split(":", 2)
    headers[key.to_s.strip.downcase] = value.to_s.strip if value
  end
  headers
end

# /ml/* an den Diarisierungs-Server durchreichen (GET /health, POST /diarize)
def proxy_ml(client, method, path, headers)
  length = headers["content-length"].to_i
  return respond(client, "413 Payload Too Large", {}) if length > MAX_BODY
  body = length.positive? ? client.read(length) : nil

  uri = URI(ML_BACKEND + path.sub(%r{\A/ml}, ""))
  request =
    case method
    when "GET"  then Net::HTTP::Get.new(uri)
    when "POST" then Net::HTTP::Post.new(uri)
    else return respond(client, "405 Method Not Allowed", {})
    end
  request["Content-Type"] = headers["content-type"] if headers["content-type"]
  request.body = body if body

  response = Net::HTTP.start(uri.hostname, uri.port,
                             open_timeout: 5, read_timeout: 1800) { |h| h.request(request) }
  respond(client, "#{response.code} #{response.message}",
          { "Content-Type" => response["Content-Type"] || "application/json" },
          response.body || "")
rescue Errno::ECONNREFUSED, Net::OpenTimeout
  respond(client, "502 Bad Gateway", { "Content-Type" => "application/json" },
          '{"detail":"ML-Server nicht erreichbar (laeuft uvicorn auf Port 8001?)"}')
end

server = TCPServer.new(BIND, PORT)
puts "Live-Audio-Analyse läuft auf http://localhost:#{PORT} (Strg+C zum Beenden)"
puts "ML-Proxy: /ml/* -> #{ML_BACKEND}"

trap("INT") { server.close; exit }

loop do
  client = server.accept
  Thread.new(client) do |c|
    request = c.gets
    next c.close if request.nil?

    method, raw_path, = request.split(" ")
    path = raw_path.to_s.split("?").first
    headers = read_headers(c)

    if path.start_with?("/ml/")
      proxy_ml(c, method, path, headers)
    else
      path = "/index.html" if path == "/"
      file = File.expand_path(File.join(ROOT, path))
      if method == "GET" && file.start_with?(ROOT + File::SEPARATOR) && File.file?(file)
        body = File.binread(file)
        type = MIME.fetch(File.extname(file), "application/octet-stream")
        respond(c, "200 OK", { "Content-Type" => type }, body)
      else
        respond(c, "404 Not Found", { "Content-Type" => "text/plain; charset=utf-8" }, "404 Not Found")
      end
    end
  rescue StandardError
    # Verbindung abgebrochen o. Ä. – ignorieren
  ensure
    c.close unless c.closed?
  end
end
