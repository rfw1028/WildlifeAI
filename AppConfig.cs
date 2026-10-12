using System.Text.Json;

namespace WildLifeAI
{
    /// <summary>
    /// Reads settings from environment variables first, then from an optional flat JSON file
    /// named appsettings.local.json (next to the executable or in the working directory).
    /// That file is git-ignored; no secrets are stored in source control.
    /// </summary>
    internal static class AppConfig
    {
        private static readonly Lazy<Dictionary<string, string>> FileValues = new(LoadFile);

        public static string? Get(string key)
        {
            var env = Environment.GetEnvironmentVariable(key);
            if (!string.IsNullOrWhiteSpace(env)) return env;
            return FileValues.Value.TryGetValue(key, out var v) && !string.IsNullOrWhiteSpace(v) ? v : null;
        }

        public static string Require(string key) =>
            Get(key) ?? throw new InvalidOperationException(
                $"Configuration value '{key}' is not set. Set it as an environment variable or in appsettings.local.json (see README.md).");

        /// <summary>Returns PREFIX_1, PREFIX_2, ... until the first missing one.</summary>
        public static string[] GetIndexed(string prefix)
        {
            var list = new List<string>();
            for (int i = 1; ; i++)
            {
                var v = Get($"{prefix}_{i}");
                if (v == null) break;
                list.Add(v);
            }
            return list.ToArray();
        }

        private static Dictionary<string, string> LoadFile()
        {
            var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var dir in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
            {
                var path = Path.Combine(dir, "appsettings.local.json");
                if (!File.Exists(path)) continue;
                using var doc = JsonDocument.Parse(File.ReadAllText(path));
                foreach (var p in doc.RootElement.EnumerateObject())
                    result[p.Name] = p.Value.ValueKind == JsonValueKind.String ? p.Value.GetString() ?? "" : p.Value.ToString();
                break;
            }
            return result;
        }
    }
}
