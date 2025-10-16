using System.IO;
using UnityEngine;

namespace SegNeshGS.Editor
{
    public class OutputPath
    {
        public static readonly string SourceDataPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "data"));
    }
}