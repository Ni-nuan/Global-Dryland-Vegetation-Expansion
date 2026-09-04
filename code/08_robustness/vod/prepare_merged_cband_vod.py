from pathlib import Path
import re
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin


# ============================================================
# 1. 需要你修改的路径
# ============================================================

# 你的 VOD nc 总文件夹
# 例如这个文件夹下面有：
#   WindSat_IB_VOD_2011
#   WindSat_IB_VOD_2012
#   GW1AM2_IB_VOD_2013
#   GW1AM2_IB_VOD_2014
#   ...
INPUT_ROOT = Path(r"VOD_nc")

# 输出 tif 文件夹
OUTPUT_DIR = Path(r"VOD_annual_max_tif")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 数据参数
# ============================================================

# 你这个 nc 文件里的 VOD 变量名
VOD_VAR = "Optical_Thickness_Nad"

# 质量控制变量
QUALITY_VAR = "Quality_Flag"

# 是否只使用 Quality_Flag == 0 的像元
# 0 表示 data OK
USE_QUALITY_FLAG = True

# 输出 nodata 值
NODATA = -9999.0

# 是否把经度 0-360 转成 -180-180
# 你上传的文件本身已经是 -180 到 180，但保留这个选项更稳
CONVERT_LON_0_360_TO_180 = True


# ============================================================
# 3. 年份识别函数
# ============================================================

def find_year_from_path(nc_path: Path):
    """
    从 nc 文件名或上级文件夹名中识别年份。

    支持：
    GW1AM2_IB_VOD_20130101_D_C_V1.nc -> 2013
    GW1AM2_IB_VOD_2013 -> 2013
    WindSat_IB_VOD_2011 -> 2011
    WindSat_IB_VOD_2012 -> 2012
    """

    # 优先从文件名中识别 YYYYMMDD
    m = re.search(r"(19\d{2}|20\d{2})\d{4}", nc_path.name)
    if m:
        return int(m.group(1))

    # 再从文件名中识别 YYYY
    m = re.search(r"(19\d{2}|20\d{2})", nc_path.name)
    if m:
        return int(m.group(1))

    # 最后从所有父文件夹名中识别 YYYY
    for part in nc_path.parts[::-1]:
        m = re.search(r"(19\d{2}|20\d{2})", part)
        if m:
            return int(m.group(1))

    return None


# ============================================================
# 4. 经纬度处理
# ============================================================

def rename_lat_lon_dims(da: xr.DataArray):
    """
    把可能的 latitude/longitude 名称统一成 lat/lon。
    当前你上传的文件已经是 lat/lon，但保留这个函数更稳。
    """

    rename_dict = {}

    for name in list(da.dims) + list(da.coords):
        low = name.lower()

        if low in ["lat", "latitude"]:
            rename_dict[name] = "lat"
        elif low in ["lon", "longitude"]:
            rename_dict[name] = "lon"

    # 避免重复 rename
    rename_dict = {
        old: new
        for old, new in rename_dict.items()
        if old != new
    }

    if rename_dict:
        da = da.rename(rename_dict)

    return da


def prepare_2d_lat_lon(da: xr.DataArray):
    """
    把 DataArray 整理成标准的二维 lat × lon。
    GeoTIFF 写出要求数组方向为：
        第一维：lat，从北到南
        第二维：lon，从西到东
    """

    da = rename_lat_lon_dims(da)
    da = da.squeeze(drop=True)

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(
            f"无法识别 lat/lon 维度。当前维度为：{da.dims}；"
            f"当前坐标为：{list(da.coords)}"
        )

    # 如果存在除 lat/lon 之外的维度，例如 time、band 等，对其取最大值
    extra_dims = [d for d in da.dims if d not in ["lat", "lon"]]
    if extra_dims:
        da = da.max(dim=extra_dims, skipna=True)

    # 转换经度 0-360 到 -180-180
    if CONVERT_LON_0_360_TO_180:
        lon = da["lon"]
        if float(lon.max()) > 180:
            da = da.assign_coords(lon=(((lon + 180) % 360) - 180))
            da = da.sortby("lon")

    # lon 从小到大，即 -180 -> 180
    if float(da["lon"][0]) > float(da["lon"][-1]):
        da = da.sortby("lon")

    # lat 从大到小，即 90 -> -90
    if float(da["lat"][0]) < float(da["lat"][-1]):
        da = da.sortby("lat", ascending=False)

    # 强制维度顺序为 lat, lon
    da = da.transpose("lat", "lon")

    return da


# ============================================================
# 5. 读取单个 nc 文件
# ============================================================

def read_one_nc_vod(nc_file: Path):
    """
    读取单个 nc 文件中的 VOD，并返回：
        arr:  二维 numpy 数组，shape = lat × lon
        lon:  经度中心点数组
        lat:  纬度中心点数组
    """

    ds = xr.open_dataset(
        nc_file,
        decode_cf=True,
        mask_and_scale=True,
        chunks=None,
        cache=False
    )

    if VOD_VAR not in ds.data_vars:
        available_vars = list(ds.data_vars)
        ds.close()
        raise ValueError(
            f"{nc_file} 中找不到变量 {VOD_VAR}。\n"
            f"当前文件变量为：{available_vars}"
        )

    da = ds[VOD_VAR]

    # 质量控制：只保留 Quality_Flag == 0 的像元
    if USE_QUALITY_FLAG and QUALITY_VAR in ds.data_vars:
        qf = ds[QUALITY_VAR]
        qf = rename_lat_lon_dims(qf)
        qf = qf.squeeze(drop=True)

        da = rename_lat_lon_dims(da)
        da = da.where(qf == 0)

    da = prepare_2d_lat_lon(da)

    arr = da.values.astype(np.float32)
    lon = da["lon"].values.astype(np.float64)
    lat = da["lat"].values.astype(np.float64)

    ds.close()

    return arr, lon, lat


# ============================================================
# 6. 构建 GeoTIFF transform
# ============================================================

def build_transform(lon, lat):
    """
    根据 lon/lat 中心点坐标构建 GeoTIFF 仿射变换。
    你这个数据是 0.25° 分辨率：
        lon center: -179.875, -179.625, ...
        lat center:  89.875,  89.625, ...
    所以 tif 边界应为：
        left=-180, right=180, top=90, bottom=-90
    """

    dx = float(abs(np.median(np.diff(lon))))
    dy = float(abs(np.median(np.diff(lat))))

    left = float(np.min(lon) - dx / 2.0)
    top = float(np.max(lat) + dy / 2.0)

    transform = from_origin(left, top, dx, dy)

    return transform


# ============================================================
# 7. 写 GeoTIFF
# ============================================================

def write_tif(out_tif: Path, arr, lon, lat):
    """
    写出单波段 float32 GeoTIFF。
    """

    transform = build_transform(lon, lat)

    arr_out = np.where(np.isfinite(arr), arr, NODATA).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": arr_out.shape[0],
        "width": arr_out.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": NODATA,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER"
    }

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr_out, 1)


# ============================================================
# 8. 逐年最大值合成
# ============================================================

def process_one_year(year, files):
    """
    对某一年所有 nc 文件做逐像元最大值合成。
    """

    print(f"\nProcessing {year}: {len(files)} files")

    annual_max = None
    ref_lon = None
    ref_lat = None

    for i, nc_file in enumerate(files, 1):
        nc_file = Path(nc_file)

        arr, lon, lat = read_one_nc_vod(nc_file)

        if annual_max is None:
            annual_max = arr
            ref_lon = lon
            ref_lat = lat
        else:
            # 检查每个文件网格是否一致
            if arr.shape != annual_max.shape:
                raise ValueError(
                    f"网格大小不一致：\n"
                    f"当前文件：{nc_file}\n"
                    f"当前 shape：{arr.shape}\n"
                    f"参考 shape：{annual_max.shape}"
                )

            if not np.allclose(lon, ref_lon) or not np.allclose(lat, ref_lat):
                raise ValueError(
                    f"经纬度坐标不一致：{nc_file}"
                )

            # np.fmax 可以跳过单侧 NaN：
            # fmax(value, NaN) = value
            # fmax(NaN, value) = value
            # fmax(NaN, NaN) = NaN
            annual_max = np.fmax(annual_max, arr)

        if i % 50 == 0 or i == len(files):
            valid_count = np.isfinite(annual_max).sum()
            print(f"  {year}: {i}/{len(files)} files processed; valid pixels = {valid_count}")

    out_tif = OUTPUT_DIR / f"VOD_{year}.tif"

    write_tif(out_tif, annual_max, ref_lon, ref_lat)

    print(f"  Saved: {out_tif}")


# ============================================================
# 9. 主程序
# ============================================================

def main():
    nc_files = sorted(INPUT_ROOT.rglob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"没有在这个文件夹下找到 nc 文件：{INPUT_ROOT}")

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
