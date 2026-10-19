using System.Buffers.Binary;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using Windows.Graphics.Imaging;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.Devices;
using Windows.Storage.Streams;

static class Program
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("VIGB");

    static async Task<int> Main(string[] args)
    {
        string? deviceName = null;
        int frames = 30;
        bool streamMode = false;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--device-name" && i + 1 < args.Length)
                deviceName = args[++i];
            else if (args[i] == "--frames" && i + 1 < args.Length)
                int.TryParse(args[++i], out frames);
            else if (args[i] == "--stream")
                streamMode = true;
        }

        if (string.IsNullOrWhiteSpace(deviceName))
        {
            await Console.Error.WriteLineAsync("Usage: VigilWindowsCapture --device-name \"Name\" [--frames N | --stream]");
            return 2;
        }

        var groups = await MediaFrameSourceGroup.FindAllAsync();
        var group = groups.FirstOrDefault(g => g.DisplayName == deviceName);
        if (group == null)
        {
            await Console.Error.WriteLineAsync($"Device not found: {deviceName}");
            foreach (var g in groups)
                await Console.Error.WriteLineAsync($"  Available: {g.DisplayName}");
            return 3;
        }

        var settings = new MediaCaptureInitializationSettings
        {
            SourceGroup = group,
            MemoryPreference = MediaCaptureMemoryPreference.Cpu,
        };

        var capture = new MediaCapture();
        await capture.InitializeAsync(settings);

        MediaFrameSource? videoSource = capture.FrameSources.Values.FirstOrDefault(s =>
            s.Info.MediaStreamType is MediaStreamType.VideoPreview or MediaStreamType.VideoRecord);
        videoSource ??= capture.FrameSources.Values.FirstOrDefault();
        if (videoSource == null)
        {
            await Console.Error.WriteLineAsync("No video source on device.");
            return 4;
        }

        using var reader = await MultiSourceMediaFrameReader.CreateForSourceGroupsAsync(
            new[] { group });
        reader.AcquisitionMode = MediaFrameReaderAcquisitionMode.Realtime;
        var status = await reader.StartAsync();
        if (status != MediaFrameReaderStartStatus.Success)
        {
            await Console.Error.WriteLineAsync($"MediaFrameReader start failed: {status}");
            return 5;
        }

        var stdout = Console.OpenStandardOutput();
        int sent = 0;
        int target = streamMode ? int.MaxValue : frames;
        var deadline = DateTime.UtcNow.AddSeconds(15);

        try
        {
            while (sent < target)
            {
                using var mediaFrame = reader.TryAcquireLatestFrame();
                if (mediaFrame?.VideoMediaFrame?.SoftwareBitmap is not SoftwareBitmap sb)
                {
                    await Task.Delay(5);
                    if (sent == 0 && DateTime.UtcNow > deadline)
                    {
                        await Console.Error.WriteLineAsync("Timeout waiting for first frame.");
                        return 6;
                    }
                    continue;
                }

                using var bgra = SoftwareBitmap.Convert(sb, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                int w = bgra.PixelWidth;
                int h = bgra.PixelHeight;
                byte[] bgraBytes = GetBitmapBytes(bgra);

                byte[] bgr = new byte[w * h * 3];
                for (int y = 0; y < h; y++)
                {
                    int srcRow = y * w * 4;
                    int dstRow = y * w * 3;
                    for (int x = 0; x < w; x++)
                    {
                        int si = srcRow + x * 4;
                        int di = dstRow + x * 3;
                        bgr[di] = bgraBytes[si];
                        bgr[di + 1] = bgraBytes[si + 1];
                        bgr[di + 2] = bgraBytes[si + 2];
                    }
                }

                await WriteFrameAsync(stdout, w, h, bgr);
                sent++;
                if (!streamMode)
                    await Task.Delay(1);
            }
        }
        finally
        {
            await reader.StopAsync();
            capture.Dispose();
        }

        return sent > 0 ? 0 : 7;
    }

    private static byte[] GetBitmapBytes(SoftwareBitmap bitmap)
    {
        int size = bitmap.PixelWidth * bitmap.PixelHeight * 4;
        byte[] data = new byte[size];
        bitmap.CopyToBuffer(data.AsBuffer());
        return data;
    }

    private static async Task WriteFrameAsync(Stream stdout, int w, int h, byte[] bgr)
    {
        byte[] header = new byte[16];
        Magic.CopyTo(header, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(4), (uint)w);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8), (uint)h);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(12), (uint)bgr.Length);
        await stdout.WriteAsync(header);
        await stdout.WriteAsync(bgr);
        await stdout.FlushAsync();
    }
}
