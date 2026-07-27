using System;
using System.IO;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Maui.Storage;

namespace Ro2ya.Services
{
    public class LocalWebServer
    {
        private HttpListener? _listener;
        private bool _isRunning;
        private int _port = 18492;
        private string? _baseWebDir;

        public string BaseUrl => $"http://127.0.0.1:{_port}/";

        public void Start()
        {
            if (_isRunning) return;

            // Extract or locate raw assets path
            _baseWebDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "wwwroot");
            if (!Directory.Exists(_baseWebDir))
            {
                // Fallback directory search if running in debug output
                var altDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Raw", "wwwroot");
                if (Directory.Exists(altDir))
                {
                    _baseWebDir = altDir;
                }
            }

            int[] portsToTry = new[] { 18492, 18493, 18494, 18495, 18496 };
            foreach (var port in portsToTry)
            {
                try
                {
                    _port = port;
                    _listener = new HttpListener();
                    _listener.Prefixes.Add(BaseUrl);
                    _listener.Start();
                    _isRunning = true;
                    Task.Run(ListenLoop);
                    break;
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Failed to start LocalWebServer on port {_port}: {ex.Message}");
                    _listener?.Close();
                    _listener = null;
                }
            }
        }

        private async Task ListenLoop()
        {
            while (_isRunning && _listener != null && _listener.IsListening)
            {
                try
                {
                    var context = await _listener.GetContextAsync();
                    _ = ProcessRequestAsync(context);
                }
                catch
                {
                    if (!_isRunning) break;
                }
            }
        }

        private async Task ProcessRequestAsync(HttpListenerContext context)
        {
            var request = context.Request;
            var response = context.Response;

            try
            {
                string rawPath = request.Url?.AbsolutePath ?? "/";
                if (rawPath == "/") rawPath = "/index.html";

                string relativePath = rawPath.TrimStart('/').Replace('/', Path.DirectorySeparatorChar);
                string localFilePath = Path.Combine(_baseWebDir ?? "", relativePath);

                byte[]? buffer = null;

                if (File.Exists(localFilePath))
                {
                    buffer = await File.ReadAllBytesAsync(localFilePath);
                }
                else
                {
                    // Attempt to read from MAUI AppPackage file
                    try
                    {
                        using var stream = await FileSystem.OpenAppPackageFileAsync($"wwwroot/{relativePath.Replace('\\', '/')}");
                        using var ms = new MemoryStream();
                        await stream.CopyToAsync(ms);
                        buffer = ms.ToArray();
                    }
                    catch
                    {
                        // File not found
                    }
                }

                if (buffer != null)
                {
                    string mimeType = GetMimeType(localFilePath);
                    response.ContentType = mimeType;
                    response.ContentLength64 = buffer.Length;
                    response.StatusCode = (int)HttpStatusCode.OK;
                    await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
                }
                else
                {
                    response.StatusCode = (int)HttpStatusCode.NotFound;
                    byte[] errBuffer = Encoding.UTF8.GetBytes("404 Not Found");
                    response.ContentLength64 = errBuffer.Length;
                    await response.OutputStream.WriteAsync(errBuffer, 0, errBuffer.Length);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error processing request: {ex.Message}");
                try
                {
                    response.StatusCode = (int)HttpStatusCode.InternalServerError;
                }
                catch { }
            }
            finally
            {
                try { response.OutputStream.Close(); } catch { }
            }
        }

        private string GetMimeType(string fileName)
        {
            string ext = Path.GetExtension(fileName).ToLowerInvariant();
            return ext switch
            {
                ".html" or ".htm" => "text/html; charset=utf-8",
                ".css" => "text/css; charset=utf-8",
                ".js" => "application/javascript; charset=utf-8",
                ".json" => "application/json; charset=utf-8",
                ".png" => "image/png",
                ".jpg" or ".jpeg" => "image/jpeg",
                ".gif" => "image/gif",
                ".svg" => "image/svg+xml",
                ".ico" => "image/x-icon",
                ".woff" => "font/woff",
                ".woff2" => "font/woff2",
                ".ttf" => "font/ttf",
                _ => "application/octet-stream",
            };
        }

        public void Stop()
        {
            _isRunning = false;
            try
            {
                _listener?.Stop();
                _listener?.Close();
            }
            catch { }
        }
    }
}
