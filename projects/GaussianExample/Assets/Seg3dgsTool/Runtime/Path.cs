using System.IO;
using UnityEngine;

namespace Seg3dgsTool.Runtime
{
    public class OutputPath
    {
        public static readonly string SourceDataPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "data"));
    }
}