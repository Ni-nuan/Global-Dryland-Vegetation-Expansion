import argparse
import os
import glob
from osgeo import gdal


def convert_and_mosaic(input_folder, output_path):
    # Step 1: 获取所有 tif 文件
    tif_list = glob.glob(os.path.join(input_folder, "*.tif"))
    if not tif_list:
        raise ValueError("❌ 文件夹中没有找到任何 .tif 文件")

    print(f"✅ 找到 {len(tif_list)} 个文件，开始统一数据类型...")

    # Step 2: 将每个 tif 转为 Float32
    float32_files = []
    for tif in tif_list:
        base = os.path.basename(tif)
        float32_tif = os.path.join(input_folder, f"f32_{base}")
        ds = gdal.Open(tif)
        gdal.Translate(
            float32_tif,
            ds,
            outputType=gdal.GDT_Float32,
            creationOptions=["TILED=YES", "COMPRESS=LZW"]
        )
        float32_files.append(float32_tif)

    print("✅ 所有文件已转换为 Float32")

    # Step 3: 使用 VRT 合并
    vrt_path = os.path.join(input_folder, "temp_mosaic.vrt")
    vrt = gdal.BuildVRT(vrt_path, float32_files)
    if vrt is None:
        raise RuntimeError("❌ VRT 构建失败")

    # Step 4: 导出最终 GeoTIFF
    gdal.Translate(
        output_path,
        vrt,
        format="GTiff",
        creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"]
    )

    print(f"✅ 拼接完成，输出文件：{output_path}")

    # Step 5: 清理临时文件
    vrt = None
    os.remove(vrt_path)
    for f in float32_files:
        if os.path.exists(f):
            os.remove(f)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GeoTIFF tiles to Float32, build a VRT, and export one mosaic GeoTIFF."
    )
    parser.add_argument("--input-folder", required=True, help="Folder containing input .tif tiles.")
    parser.add_argument("--output", required=True, help="Output mosaic GeoTIFF path.")
    args = parser.parse_args()
    convert_and_mosaic(args.input_folder, args.output)


if __name__ == "__main__":
    main()

