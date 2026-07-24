#!/usr/bin/env ruby
# Kleiner statischer Server für das Live-Audio-Analyse-Tool.
# Nur Ruby-Stdlib (socket) – keine Gems nötig.
#
#   ruby live-audio-tool/server.rb [PORT]
#
# Danach im Browser http://localhost:8000 öffnen (Chrome/Edge empfohlen).
# Mikrofonzugriff erfordert einen "secure context" – localhost zählt dazu.

require "socket"

PORT = (ARGV[0] || 8000).to_i
ROOT = File.expand_path(__dir__)

MIME = {
  ".html" => "text/html; charset=utf-8",
  ".js"   => "text/javascript; charset=utf-8",
  ".css"  => "text/css; charset=utf-8",
  ".json" => "application/json",
  ".svg"  => "image/svg+xml",
  ".png"  => "image/png",
  ".ico"  => "image/x-icon",
}.freeze

def respond(client, status, headers, body = "")
  client.write "HTTP/1.1 #{status}\r\n"
  headers.merge("Content-Length" => body.bytesize, "Connection" => "close").each do |k, v|
    client.write "#{k}: #{v}\r\n"
  end
  client.write "\r\n"
  client.write body
end

server = TCPServer.new("127.0.0.1", PORT)
puts "Live-Audio-Analyse läuft auf http://localhost:#{PORT} (Strg+C zum Beenden)"

trap("INT") { server.close; exit }

loop do
  client = server.accept
  Thread.new(client) do |c|
    request = c.gets
    next c.close if request.nil?

    # restliche Header verwerfen
    nil while (line = c.gets) && line != "\r\n"

    path = request.split(" ")[1].to_s.split("?").first
    path = "/index.html" if path == "/"
    file = File.expand_path(File.join(ROOT, path))

    if file.start_with?(ROOT + File::SEPARATOR) && File.file?(file)
      body = File.binread(file)
      type = MIME.fetch(File.extname(file), "application/octet-stream")
      respond(c, "200 OK", { "Content-Type" => type }, body)
    else
      respond(c, "404 Not Found", { "Content-Type" => "text/plain; charset=utf-8" }, "404 Not Found")
    end
  rescue StandardError
    # Verbindung abgebrochen o. Ä. – ignorieren
  ensure
    c.close unless c.closed?
  end
end
