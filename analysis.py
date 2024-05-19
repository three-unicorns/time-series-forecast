#%%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import cfgrib

lon_min = -20
lon_max = 45
lat_min = 34
lat_max = 60

# lon_min = 20.9
# lon_max = 26.8
# lat_min = 53.9
# lat_max = 56.4



def plot_dataset(wind : xr.Dataset, temp : xr.DataArray = None, vmax=40, vmin=-40, cmap="viridis", title=None):
    projection = ccrs.Mercator()
    crs = ccrs.PlateCarree()
    ax = plt.axes(projection=projection, frameon=True)
    gl = ax.gridlines(crs=crs, draw_labels=True,
                    linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
    gl.xlabel_style = {"size" : 7}
    gl.ylabel_style = {"size" : 7}
    import cartopy.feature as cf
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)
    if temp is not None:
          temp.plot(x="longitude", y="latitude", ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    if wind is not None:
        wind.plot.quiver(x="longitude", y="latitude", u="u10", v="v10", ax=ax, width = 0.0007, transform=ccrs.PlateCarree())


    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)
    plt.title(title)
    plt.draw()
#%%
grib_data = cfgrib.open_datasets("/workspace/20240501-1200-panguweather-5-steps.grib")
wind = grib_data[0]
temp = grib_data[1]
# %%
from copy import deepcopy
from cartopy.util import add_cyclic_point

# Create copy of the data so we maintain original data
crop_wind = deepcopy(wind)
crop_temp = deepcopy(temp)

def adjust_longitude(dataset: xr.Dataset) -> xr.Dataset:
        """Swaps longitude coordinates from range (0, 360) to (-180, 180)
        Args:
            dataset (xr.Dataset): xarray Dataset
        Returns:

            xr.Dataset: xarray Dataset with swapped longitude dimensions
        """
        lon_name = "longitude"  # whatever name is in the data

        # Adjust lon values to make sure they are within (-180, 180)
        dataset["_longitude_adjusted"] = xr.where(
            dataset[lon_name] > 180, dataset[lon_name] - 360, dataset[lon_name]
        )
        dataset = (
            dataset.swap_dims({lon_name: "_longitude_adjusted"})
            .sel(**{"_longitude_adjusted": sorted(dataset._longitude_adjusted)})
            .drop(lon_name)
        )

        dataset = dataset.rename({"_longitude_adjusted": lon_name})
        return dataset

crop_wind = adjust_longitude(crop_wind)
crop_temp = adjust_longitude(crop_temp)

crop_wind = crop_wind.sel(
        latitude=slice(lat_max, lat_min, 3),
        longitude=slice(lon_min, lon_max, 3),
    )
crop_temp = crop_temp.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
    )

crop_temp["t2m"] -= 273

# %%
# grib_data = cfgrib.open_datasets("/workspace/real.grib")
# ds = grib_data[0]
# crop_temp = deepcopy(ds)
# crop_temp = adjust_longitude(crop_temp)
# crop_temp = crop_temp.sel(
#         latitude=slice(lat_max, lat_min),
#         longitude=slice(lon_min, lon_max),
#     )
# crop_temp["t2m"] -= 273
# %%
def plotTempDifferences(temp1 : xr.Dataset, temp2 :xr.Dataset, step=0, **kwargs):
    title = f"Temperature difference at step {step} (PanguWeather)"
    diff = temp1.isel(step=step)["t2m"] - temp2.isel(step=step)["t2m"]
    plot_dataset(None, diff, title=title, **kwargs)
#%%
def update(frame):
     fig.clf()
     title = f"Data at step={frame} (ERA5 Real Data)"
     plot_dataset(crop_wind.isel(step=frame), crop_temp.isel(step=frame)["t2m"], title=title)
#%%
from matplotlib.animation import FuncAnimation
fig = plt.figure(figsize=(16,9), dpi=150) 

ani = FuncAnimation(fig, update, frames=6)
ani.save("/workspace/real.gif", fps=1)
#%%
grib_data = cfgrib.open_datasets("/workspace/real.grib")
ds = grib_data[0]
real = deepcopy(ds)
real = adjust_longitude(real)
real = real.sel(
        latitude=slice(lat_max, lat_min, 3),
        longitude=slice(lon_min, lon_max, 3),
    )
real["t2m"] -= 273
real = real.drop("step").rename({"time": "step"})
real = real.drop_isel(step=[0,1])
plotTempDifferences(crop_temp, real, step=3, vmin=-5, vmax=5, cmap="coolwarm")
# %%
def update_diff(frame):
     fig.clf()
     plotTempDifferences(crop_temp, real, step=frame, vmin=-5, vmax=5, cmap="coolwarm")

fig = plt.figure(figsize=(16,9), dpi=150) 
ani = FuncAnimation(fig, update_diff, frames=6)
ani.save("/workspace/pangu-diff.gif", fps=1)
# %%
diff = real["t2m"] - crop_temp["t2m"]
diff *= diff
# %%
import numpy as np
arr = {}
for i in range(1,6):
    r = np.array(real.isel(step=i).t2m)
    f = np.array(crop_temp.isel(step=i).t2m)
    err = r-f
    err *= err
    rmse = err.mean()**(1/2)
    arr[i]=rmse
plt.bar(x=arr.keys(), height=arr.values())
plt.title("Temperature RMSE for each step (PanguWeather)")
# print(f"Temperature RMSE: {rmse}")