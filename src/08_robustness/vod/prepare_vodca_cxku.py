from pathlib import Path
import re
import numpy as np
import h5py
import rasterio
from rasterio.transform import from_origin


# ============================================================
# 1. 修改这里：输入和输出路径
# ============================================================

# VODCA_CXKu_daily 总文件夹
# 例如：
# VODCA_CXKu_daily/
#   2000/
#     daily_images_2000-01-01.nc
#     daily_images_2000-01-02.nc
#   2001/
#     ...
INPUT_ROOT = Path(r"VODCA_CXKu_daily")

# 年最大值 tif 输出文件夹
OUTPUT_DIR = Path(r"VODCA_CXKu_tif")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 数据参数
# ============================================================

# VODCA CXKu 变量名
VOD_VAR = "VODCA_CXKu"

# 输出 NoData
OUT_NODATA = -9999.0

# 是否使用 nc 中的 valid_range 属性筛选有效值
USE_VALID_RANGE = True

# 输出文件名前缀
OUT_PREFIX = "VODCA_CXKu"


# ============================================================
# 3. 从文件名或文件夹名识别年份
# ============================================================

def find_year_from_path(nc_path: Path):
    """
    支持：
    daily_images_2000-01-01.nc -> 2000
    VODCA_CXKu_daily/2000/daily_images_2000-01-01.nc -> 2000
    """

    # 优先从文件名识别年份
    m = re.search(r"(19\d{2}|20\d{2})", nc_path.name)
    if m:
        return int(m.group(1))

    # 再从父文件夹识别年份
    for part in nc_path.parts[::-1]:
        m = re.fullmatch(r"(19\d{2}|20\d{2})", part)
        if m:
            return int(m.group(1))

    # 最后从任意路径片段识别年份
    for part in nc_path.parts[::-1]:
        m = re.search(r"(19\d{2}|20\d{2})", part)
        if m:
            return int(m.group(1))

    return None


# ============================================================
# 4. 读取单个 VODCA nc 文件
# ============================================================

def read_one_vodca_nc(nc_file: Path):
    """
    读取一个 daily_images_YYYY-MM-DD.nc 文件。

    返回：
        arr: 2D numpy array, shape = lat × lon
        lon: 1D longitude center coordinates
        lat: 1D latitude center coordinates
    """

    with h5py.File(nc_file, "r") as f:

        if VOD_VAR not in f:
            raise ValueError(
                f"{nc_file} 中找不到变量 {VOD_VAR}。"
                f"当前变量包括：{list(f.keys())}"
            )

        arr = f[VOD_VAR][:]
        lat = f["lat"][:].astype(np.float64)
        lon = f["lon"][:].astype(np.float64)

        var_attrs = f[VOD_VAR].attrs

        fill_value = var_attrs.get("_FillValue", None)
        valid_range = var_attrs.get("valid_range", None)

    # 原始 shape 是 time × lat × lon，例如 (1, 720, 1440)
    arr = np.asarray(arr)

    # 如果有 time 维，先对 time 取最大值
    if arr.ndim == 3:
        # 通常是 time, lat, lon
        arr = np.nanmax(arr, axis=0)
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(
            f"{nc_file} 的 {VOD_VAR} 维度异常，shape = {arr.shape}"
        )

    arr = arr.astype(np.float32)

    # 处理 FillValue
    if fill_value is not None:
        fill_value = np.asarray(fill_value).ravel()[0]
        arr[arr == fill_value] = np.nan

    # 处理 valid_range，例如 [0, 4]
    if USE_VALID_RANGE and valid_range is not None:
        valid_range = np.asarray(valid_range).ravel()
        vmin = float(valid_range[0])
        vmax = float(valid_range[1])
        arr[(arr < vmin) | (arr > vmax)] = np.nan

    # lon 如果不是从西到东，则排序
    if lon[0] > lon[-1]:
        idx = np.argsort(lon)
        lon = lon[idx]
        arr = arr[:, idx]

    # 如果经度是 0–360，则转成 -180–180 并重排
    if np.nanmax(lon) > 180:
        lon_new = ((lon + 180) % 360) - 180
        idx = np.argsort(lon_new)
        lon = lon_new[idx]
        arr = arr[:, idx]

    # lat 如果不是从北到南，则排序成北到南
    if lat[0] < lat[-1]:
        idx = np.argsort(lat)[::-1]
        lat = lat[idx]
        arr = arr[idx, :]

    return arr, lon, lat


# ============================================================
# 5. 构建 GeoTIFF transform
# ============================================================

def build_transform(lon, lat):
    """
    lon/lat 是像元中心点坐标。

    对该数据：
        lon: -179.875, -179.625, ...
        lat:  89.875,   89.625, ...

    因此 GeoTIFF 左上角边界为：
        left = -180
        top  = 90
    """

    dx = float(abs(np.median(np.diff(lon))))
    dy = float(abs(np.median(np.diff(lat))))

    left = float(np.min(lon) - dx / 2.0)
    top = float(np.max(lat) + dy / 2.0)

    return from_origin(left, top, dx, dy)


# ============================================================
# 6. 写 GeoTIFF
# ============================================================

def write_geotiff(out_tif: Path, arr, lon, lat):
    """
    写出单波段 float32 GeoTIFF。
    """

    transform = build_transform(lon, lat)

    arr_out = np.where(np.isfinite(arr), arr, OUT_NODATA).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": arr_out.shape[0],
        "width": arr_out.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": OUT_NODATA,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER"
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr_out, 1)


# ============================================================
# 7. 逐年最大值合成
# ============================================================

def process_one_year(year: int, files):
    """
    对某一年所有 daily nc 文件做逐像元最大值合成。
    """

    print(f"\nProcessing {year}: {len(files)} files")

    annual_max = None
    ref_lon = None
    ref_lat = None

    for i, nc_file in enumerate(files, 1):
        nc_file = Path(nc_file)

        arr, lon, lat = read_one_vodca_nc(nc_file)

        if annual_max is None:
            annual_max = arr
            ref_lon = lon
            ref_lat = lat
        else:
            if arr.shape != annual_max.shape:
                raise ValueError(
                    f"网格大小不一致：\n"
                    f"当前文件：{nc_file}\n"
                    f"当前 shape：{arr.shape}\n"
                    f"参考 shape：{annual_max.shape}"
                )

            if not np.allclose(lon, ref_lon):
                raise ValueError(f"经度坐标不一致：{nc_file}")

            if not np.allclose(lat, ref_lat):
                raise ValueError(f"纬度坐标不一致：{nc_file}")

            # np.fmax 可以忽略单侧 NaN：
            # fmax(value, NaN) = value
            # fmax(NaN, value) = value
            # fmax(NaN, NaN) = NaN
            annual_max = np.fmax(annual_max, arr)

        if i % 50 == 0 or i == len(files):
            valid_pixels = int(np.isfinite(annual_max).sum())
            print(
                f"  {year}: {i}/{len(files)} files processed; "
                f"valid pixels = {valid_pixels}"
            )

    out_tif = OUTPUT_DIR / f"{OUT_PREFIX}_{year}.tif"

    write_geotiff(out_tif, annual_max, ref_lon, ref_lat)

    print(f"  Saved: {out_tif}")


# ============================================================
# 8. 主程序
# ============================================================

def main():
    nc_files = sorted(INPUT_ROOT.rglob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"没有在该文件夹下找到 nc 文件：{INPUT_ROOT}")

    files_by_year = {}

    for nc_file in nc_files:
        year = find_year_from_path(nc_file)

        if year is None:
            print(f"Warning: 无法识别年份，跳过：{nc_file}")
            continue

        files_by_year.setdefault(year, []).append(nc_file)

    if not files_by_year:
        raise RuntimeError("没有任何 nc 文件成功识别出年份。")

    print("Detected years:")
    for year in sorted(files_by_year):
        print(f"  {year}: {len(files_by_year[year])} files")

    for year in sorted(files_by_year):
        process_one_year(year, files_by_year[year])

    print("\nAll done.")


if __name__ == "__main__":
    main()