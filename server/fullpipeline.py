#vanila
import train_scene

#saga
import extract_segment_everything_masks
import get_scale
import train_saga
import saga_gui

#sugar
import clipsam.faseClipExtract
import train_sugar
import sys
import argparse
import os


import time
import sys
import traceback
import os 

def safe_run(name, func, args):
    start_time = time.time()
    print(f"[{name}] Starting...")
    
    success = False
    res = None
    try:
        res = func(args)
        success = True
    except Exception as e:
        print(f"[{name}] Failed: {e}")
        traceback.print_exc()
    finally:
        end_time = time.time()
        duration = end_time - start_time
        minutes, seconds = divmod(duration, 60)
        
        if success:
            print(f"[{name}] Completed in {int(minutes)}m {seconds:.2f}s.\n")
        else:
            print(f"[{name}] Process terminated after {int(minutes)}m {seconds:.2f}s.\n")

    if not success:
        sys.exit(1)
    
    return res


def SegMeshGS_full_pipeline(colmap_folder, downsample=2, iterations=10000, num_sampled_rays=1000):
    
    # ==============    vanilla    ================
    safe_run("Vanilla Training", train_scene.main, ["-s", colmap_folder])
    
    
    # ==============      SAGA     ==================
    safe_run("SAGA::SAMExtraction", extract_segment_everything_masks.main, [
        "--image_root", colmap_folder,
        "--downsample", str(downsample)
    ])
    safe_run("SAGA::GetScale", get_scale.main, [
        "--image_root", colmap_folder,
        "--model_path", colmap_folder+"/saga"
    ])
    safe_run("SAGA::Train", train_saga.main, [
        "-m", colmap_folder + "/saga",
        "--iterations" , iterations,
        '--num_sampled_rays', num_sampled_rays
    ])
    
    segment_file_name = safe_run("SAGA::GUI", saga_gui.main, [
        "-m", colmap_folder + "/saga"
    ])
    
    print(segment_file_name)
    
    # ================      SuGaR      ===================
    safe_run("SuGaR::faseClipExtract", clipsam.faseClipExtract.main, [
        "--input_folder", colmap_folder,
        "--query", segment_file_name
    ])
    
    safe_run("SuGaR::Train", train_sugar.main, [
        "-s", colmap_folder,
        '-r', 'dn_consistency',
        '--high_poly', 'True',
        '--export_obj', 'True',
        '--gs_output_dir', os.path.join(colmap_folder, "saga"),
        '--white_background', 'TRUE',
        '--segment_targetname', segment_file_name, 
    ])
    
    return 0
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full 3DGS → SAGA → SUGAR pipeline")
    parser.add_argument("--colmap_folder", required=True, help="colmap folder path")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num_sampled_rays", type=int, default=1000)

    args = parser.parse_args()

    res = SegMeshGS_full_pipeline(
        args.colmap_folder,
        args.downsample,
        args.iterations,
        args.num_sampled_rays
    )
    
     # All done
    if res == 0:
        print("SegMeshGS complete.")
    
    