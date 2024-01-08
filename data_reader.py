"""
    Thomas DeWitt
    Functions for reading and formatting satellite data.
"""

# from goes2go.data import goes_nearesttime
import numpy as np
import os
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import gzip
import glob
import datetime
from pyhdf.error import HDF4Error
from warnings import warn
from pyproj import Proj
from numba import njit
import time
import matplotlib.pyplot as plt
# from geographiclib.polygonarea import PolygonArea
# from geographiclib.geodesic import Geodesic
# import plotting_functions
# from skimage.measure import find_contours
from global_land_mask import is_land
from psutil import Process

# import  n2str
import samQVSAT
from PyThomas import useful_functions as uf
from directories import StringDirectories
dirs = StringDirectories()
ask_to_delete = True   # Whether to delete a corrupted file if one is found
degree_per_rad = 180/np.pi
r_Earth = 6371    # arithmetic mean
def main():
    start = time.time()
    
    # fig, axes = plt.subplots(2,3, figsize=(6,6))
    # for ax in np.array(axes).flatten(): 
    #     ax.xaxis.set_visible(False)
    #     ax.yaxis.set_visible(False)
    #     ax.invert_yaxis()

    # s = MODIS_Homebrew_filenames([str(y) for y in range(2021001, 2021011)], print_fnames=True, include_image=False)[0]

    smash_times = ['0000041400','0000043200']
    s = SAM_filenames(smash_times,'GATE_IDEAL',slices=True, threshold=.03)[60]

    s.load_files()
    
    # print(len(s))
    # s.load_files()
    # axes[0][0].pcolormesh(s.lat)
    # axes[0][0].set_title('Lat')
    # axes[1][0].pcolormesh(s.lon)
    # axes[1][0].set_title('Lon')

    # axes[0][1].pcolormesh(s.cloud_mask, cmap=plotting_functions.cloud_colors)
    # axes[0][1].set_title(f'Cloud Mask\nThresh: {s.thresh}')
    # axes[0][2].imshow(s.image)
    # axes[0][2].set_title('RGB Image')
    # axes[1][1].pcolormesh(s.sun_glint)
    # axes[1][1].set_title(f'Sun Glint')
    # axes[1][2].pcolormesh(s.sensor_azimuth)
    # axes[1][2].set_title('Sensor Azimuth')

    # plt.tight_layout()
    # plotting_functions.savefig('Hour 1900', )

    print('Total time {:.01f}s ({:.01f}m)'.format(time.time()-start, (time.time()-start)/60))

class Scene():
    def __init__(self, file_filename, satellite_name):
        """
            Input:
                file_filename: Name of original data file
                satellite_name: 
                    Currently supported: 
                        ['VIIRS','GEORING', 'POLDER','MODIS 1km', 'MODIS 5km','MODIS Homebrew', 'GOES WEST', 'GOES EAST','HIMAWARI+1407','MSG+0000','MSG+0415','EPIC']
        """
        if satellite_name == 'SAM': raise ValueError('Use SamScene object for SAM')
        self.file_filename = file_filename    # Data fname. string or list of strings if combining files
        self.satellite_name = satellite_name
        self.altitude = None
        self.nadir_resolution = None
        self.sensor_zenith_max = None
        self.lat_range = None   # None if no restriction, otherwise (minlat, maxlat)
        self.land_sea_truncation = None    # None if no restiction, 0 if sea only, 1 if land only

        # Data arrays:
        self.shape = None
        self.cloud_mask = None
        self.lon = None
        self.lat = None
        self.day_night = None   # 0 is Night, 1 is Day
        self.land_sea = None    # 0 = Sea, 1 = Land. Only calculated if self.land_sea_truncation is not None
        self.sensor_zenith = None
        self.sensor_azimuth = None
        self.date = None           # Sometimes an array, sometimes just a value
        self.sun_glint = None
        self.optical_depth = None
        self.pixel_sizes_x, self.pixel_sizes_y = None, None


    def get_pixel_sizes(self):
        if self.satellite_name in ['GEORING', 'POLDER']:
            self.pixel_sizes_x, self.pixel_sizes_y = get_gridded_pixel_sizes(self.resolution_degrees, self.lat)
        else:
            if self.satellite_name in ['MODIS 1km', 'MODIS 5km','MODIS Homebrew', 'GOES WEST', 'GOES EAST','HIMAWARI+1407','MSG+0000','MSG+0415','EPIC']:
                increase_pixels = 1
            elif self.satellite_name in ['VIIRS']:
                increase_pixels = 0
            self.pixel_sizes_x, self.pixel_sizes_y = get_pixel_sizes(self.nadir_resolution, self.sensor_zenith, self.sensor_azimuth, self.altitude, increase_pixels)
            self.cloud_mask[np.isnan(self.pixel_sizes_x)] = np.nan
            self.cloud_mask[np.isnan(self.pixel_sizes_y)] = np.nan

    def truncate_scene_area(self):
        if self.sensor_zenith_max is not None:
            self.cloud_mask = truncate_image_by_sensor_zenith(self.cloud_mask, self.sensor_zenith, self.sensor_zenith_max)
            self.pixel_sizes_x = truncate_image_by_sensor_zenith(self.pixel_sizes_x, self.sensor_zenith, self.sensor_zenith_max)
            self.pixel_sizes_y = truncate_image_by_sensor_zenith(self.pixel_sizes_y, self.sensor_zenith, self.sensor_zenith_max)
        if self.lat_range is not None:
            self.cloud_mask = truncate_image_by_latitude(self.cloud_mask, self.lat, *self.lat_range)
            self.pixel_sizes_x = truncate_image_by_latitude(self.pixel_sizes_x, self.lat, *self.lat_range)
            self.pixel_sizes_y = truncate_image_by_latitude(self.pixel_sizes_y, self.lat, *self.lat_range)
        if self.land_sea_truncation is not None:
            lat_nonan = self.lat.copy()
            lon_nonan = self.lon.copy()
            lat_nonan[np.isnan(lat_nonan)] = 0
            lon_nonan[np.isnan(lon_nonan)] = 0
            self.land_sea = is_land(lat_nonan,lon_nonan).astype(np.float16)
            self.land_sea[np.isnan(self.lat) | np.isnan(self.lon)] = np.nan

            self.cloud_mask = truncate_image_by_surface(self.cloud_mask, self.land_sea, self.land_sea_truncation)
            self.pixel_sizes_x = truncate_image_by_surface(self.pixel_sizes_x, self.land_sea, self.land_sea_truncation)
            self.pixel_sizes_y = truncate_image_by_surface(self.pixel_sizes_y, self.land_sea, self.land_sea_truncation)

    def load_files(self):
        if self.satellite_name == 'GOES WEST' or self.satellite_name == 'GOES EAST' or self.satellite_name == 'MSG+0415' or self.satellite_name == 'MSG+0000' or self.satellite_name == 'HIMAWARI+1407':
            self = _GEOST_read(self)
        elif self.satellite_name == 'MODIS 5km':
            self = _MODIS_5km_read(self, self.cloud_thresh)
        elif self.satellite_name == 'MODIS 1km':
            self = _MODIS_1km_read(self)
        elif self.satellite_name == 'MODIS Homebrew':
            self = _MODIS_Homebrew_read(self)
        elif self.satellite_name == 'POLDER':
            self = _POLDER_read(self)
        elif self.satellite_name == 'EPIC':
            self = _EPIC_read(self)
        elif self.satellite_name == 'VIIRS':
            self = _VIIRS_read(self)
        elif self.satellite_name == 'GEORING':
            self = _GEORING_read(self)

        self.get_pixel_sizes()
        self.truncate_scene_area()


class SamScene(Scene):
    def __init__(self, filename, run, threshold, slice=None):
        """
            run: 'TWPICE_100m', 'TWPICE_800m', 'RCE295'
            slice: 
                None: Cloud mask vertically summed
                int: [0,255]: Cloud mask only taken from that vertical level
            threshold: 
                if slice is None: g/kg, Non-precipitating Condensate (Water+Ice) 
                else: optical depth calculated from cloud ice, cloud water, and snow
        """

        self.file_filename = filename
        self.run = run
        self.satellite_name = run
        self.threshold = threshold
        self.slice = slice
        self.timestamp = None

        self.sensor_zenith_max = None
        self.lat_range = None
        self.land_sea_truncation = None

        if not run in ['TWPICE.RADSNOW.800m','TWPICE.100m','RCE295','GATE_IDEAL']: raise ValueError('Run {} unknown'.format(run))

    def load_files(self): self = _SAM_read(self)

    def get_3d_cloud_mask(self, first_only = False): 
        if self.threshold >= 1: warn('Using threshold of {} g/kg for each grid box which is very high'.format(self.threshold))
        self = _SAM_get_3d_cloudmask(self, first_only = first_only)

        


def MODIS_5km_filenames(dates, n_files_to_combine=18, cloud_thresh=50, time=None, sensor_zenith_max = 60, suppress_missing_warning=True, lat_range=None, land_sea=None):
    """
        Input:
            dates - Dates to look for data: list or str, format yyyyddd
            cloud_thresh - Threshold to consider cloudy. Clouds are >= this: int, percent
            n_files_to_combine - Combine this many files into one Scene: int 
                                    must be multiple of 288 (there are 288 files per day)
            time - If fewer than all files for each day are desired, specify the time(s) desired: str or list
            sensor_zenith_max: None or degrees. Mark all data outside this as nan.
            suppress_missing_warning - when file is missing, print info about it: bool
        Output:
            List of Scene objects
                Special attributes for Scenes:
                    - n_files_to_combine
    """
    if type(time) == str: time = [time]
    if type(dates) == str: dates = [dates]
    assert 288/n_files_to_combine == 288//n_files_to_combine, "n_files_to_combine must be multiple of 288"
    all_scenes = []
    if time is None:
        all_hours = []
        for hour in range(24):
            for minute in range(0,60,5):
                all_hours.append(datetime.time(hour=hour, minute=minute).strftime('%H%M'))
    else:
        if len(time) == 1 and n_files_to_combine > 1: n_files_to_combine = 1
        all_hours = time


    for date in dates:
        folder_name = datetime.datetime.strptime(date, "%Y%j").strftime("%Y/%Y_%m_%d")
        for n_chunk in range(288 // n_files_to_combine):
            if n_chunk >= len(all_hours):
                break
            times = range(n_chunk * n_files_to_combine, (n_chunk + 1) * n_files_to_combine)
            filenames = []
            omit = False
            for hour in times:
                time = all_hours[hour]
                file = glob.glob(dirs.mod06_dir + folder_name + '/MOD06_L2.A' + date + '.' + time + '*.hdf')
                if len(file) == 0:
                    if not suppress_missing_warning:
                        print("Missing file at "+folder_name+' '+time+', omitting this orbit')
                    omit = True
                    break
                assert len(file) == 1, "More than one file available for "+folder_name+' '+time
                filenames.append(file[0])
            if not omit:
                new_scene = Scene(filenames, 'MODIS 5km')
                new_scene.n_files_to_combine = n_files_to_combine
                new_scene.altitude = 705 # km
                new_scene.nadir_resolution = 5 #km
                new_scene.sensor_zenith_max = sensor_zenith_max
                new_scene.cloud_thresh = cloud_thresh
                new_scene.land_sea_truncation = land_sea
                new_scene._filename = '{}_Scene_{}_{}_{}_{}_{}'.format('MODIS 5km', sensor_zenith_max, date, times, n_files_to_combine, cloud_thresh).replace(' ','_').replace('/','_')+'/'
                new_scene.lat_range = lat_range
                all_scenes.append(new_scene)
    return all_scenes
def MODIS_1km_filenames(dates, n_files_to_combine=18, time=None, sensor_zenith_max = 60, suppress_missing_warning=True, lat_range=None, land_sea=None, tau_thresh=None):
    """
        Input:
            dates - Dates to look for data: list or str, format yyyyddd
            cloud_thresh -      inclusive
            n_files_to_combine - Combine this many files into one Scene: int 
                                    must be multiple of 288 (there are 288 files per day)
            time - If fewer than all files for each day are desired, specify the time(s) desired: str or list
            sensor_zenith_max: None or degrees. Mark all data outside this as nan.
            suppress_missing_warning - when file is missing, print info about it: bool
            tau_thresh - if other than None, use optical depth thresh instead of premade cloud mask
        Output:
            List of Scene objects
                Special attributes for Scenes:
                    - n_files_to_combine
    """
    if type(time) == str: time = [time]
    if type(dates) == str: dates = [dates]
    assert 288/n_files_to_combine == 288//n_files_to_combine, "n_files_to_combine must be multiple of 288"
    all_scenes = []
    if time is None:
        all_hours = []
        for hour in range(24):
            for minute in range(0,60,5):
                all_hours.append(datetime.time(hour=hour, minute=minute).strftime('%H%M'))
    else:
        if len(time) == 1 and n_files_to_combine > 1: n_files_to_combine = 1
        all_hours = time


    for date in dates:
        folder_name = datetime.datetime.strptime(date, "%Y%j").strftime("%Y/%Y_%m_%d")
        for n_chunk in range(288 // n_files_to_combine):
            if n_chunk >= len(all_hours):
                break
            times = range(n_chunk * n_files_to_combine, (n_chunk + 1) * n_files_to_combine)
            filenames = []
            omit = False
            for hour in times:
                time = all_hours[hour]
                # Need to read both level 3 and level 6
                # print(dirs.mod03_dir + folder_name + '/MOD03.A' + date + '.' + time + '*.hdf')
                mod03_file = glob.glob(dirs.mod03_dir + folder_name + '/MOD03.A' + date + '.' + time + '*.hdf')
                mod06_file = glob.glob(dirs.mod06_dir + folder_name + '/MOD06_L2.A' + date + '.' + time + '*.hdf')
                if len(mod03_file) == 0 or len(mod06_file) == 0:
                    if not suppress_missing_warning:
                        print("Missing file at "+folder_name+' '+time+', omitting this orbit')
                    omit = True
                    break
                assert len(mod03_file) == 1, "More than one file available for "+folder_name+' '+time
                filenames.append((mod03_file[0], mod06_file[0]))
            if not omit:
                new_scene = Scene(filenames, 'MODIS 1km')
                new_scene.n_files_to_combine = n_files_to_combine
                new_scene.altitude = 705 # km
                new_scene.nadir_resolution = 1 #km
                new_scene.sensor_zenith_max = sensor_zenith_max
                new_scene.date = datetime.datetime.strptime(date, "%Y%j")
                new_scene.lat_range = lat_range
                new_scene.land_sea_truncation = land_sea
                new_scene.tau_thresh = tau_thresh
                new_scene._filename = '{}_Scene_{}_{}_{}_{}'.format('MODIS 1km', sensor_zenith_max, date, times, n_files_to_combine).replace(' ','_').replace('/','_')+'/'
                all_scenes.append(new_scene)
    return all_scenes
def MODIS_Homebrew_filenames(dates, hours = ['1900','1905'], sensor_zenith_max = 60, thresh=0.5, suppress_missing_warning=False, include_image=False, print_fnames=False):
    """
        Input:
            dates - Dates to look for data: list or str, format yyyyddd
            thresh - between 0 and 1 - normalized radiance threshold
        Output:
            List of Scene objects
                Special attributes for Scenes:
                    - thresh - between 0 and 1 - normalized radiance threshold
                    - include_image: T/F: whether to save image as attr for scene
                    - image: RGB image
                    - sun_glint: 1 if Yes, 0 if No
    """
    if type(dates) == str: dates = [dates]
    
    
    all_scenes = []
    all_fnames = []

    for date in dates:
        folder_name = datetime.datetime.strptime(date, "%Y%j").strftime("%Y/%Y_%m_%d")
        # Get hours by just finding what is available in filesystem
        hours = [hour.split('.061.')[0][-4:]for hour in os.listdir(dirs.mod02_dir + folder_name) if hour[-4:] == '.hdf' ]
        for hour in hours:

            mod06_file = glob.glob(dirs.mod06_dir + folder_name + '/MOD06_L2.A' + date + '.' + hour + '*.hdf')
            mod03_file = glob.glob(dirs.mod03_dir + folder_name + '/MOD03.A' + date + '.' + hour + '*.hdf')
            mod02_file = glob.glob(dirs.mod02_dir + folder_name + '/MOD02QKM.A' + date + '.' + hour + '*.hdf')
            mod021_file = glob.glob(dirs.mod021_dir + folder_name + '/MOD021KM.A' + date + '.' + hour + '*.hdf')
            if len(mod03_file) == 0:
                if not suppress_missing_warning:
                    print("Missing MOD03 file at "+folder_name+' '+hour+', omitting this granule')
                continue
            if len(mod02_file) == 0:
                if not suppress_missing_warning:
                    print("Missing MOD02 file at "+folder_name+' '+hour+', omitting this granule')
                continue
            if len(mod021_file) == 0:
                if include_image:   # only need for image
                    if not suppress_missing_warning:
                        print("Missing MOD021 file at "+folder_name+' '+hour+', omitting this granule')
                    continue
                else: mod021_file = [None]
            all_fnames.append(mod02_file[0])
            new_scene = Scene([mod02_file[0], mod03_file[0], mod06_file[0], mod021_file[0]], 'MODIS Homebrew')
            new_scene.altitude = 705 # km
            new_scene.nadir_resolution = .25 #km
            new_scene.thresh = thresh
            new_scene.include_image = include_image
            new_scene.sensor_zenith_max = sensor_zenith_max
            new_scene.date = datetime.datetime.strptime(date+hour, "%Y%j%H%M")
            new_scene._filename = '{}_Scene_{}_{}_{}_{}'.format('MODIS Homebrew', sensor_zenith_max, thresh, date, hour).replace(' ','_').replace('/','_')+'/'
            all_scenes.append(new_scene)

    if print_fnames:
        print('-'*50,f'\n    MODIS Homebrew Filenames ({len(all_fnames)} total)')
        [print(f'   {fname}') for fname in all_fnames]
        print('-'*50)

    return all_scenes
def GEOST_filenames(dates, 
                    nearest_time=None, 
                    sensor_zenith_max = 60,
                    satellite='GOES WEST', lat_range=None, land_sea=None,
                    cloud_temp_thresh = None):
    """
        Input:
            dates - Dates to look for data: list or str, format yyyyddd
            nearest_time - If a specific time is desired. 
                            Can be %H%M, 'day', or 'night': then will examine one image
                            If None, examine 'day' and 'night' (noon and midnight)

            one_per_day - Return only one image: bool
            cloud_temp_thresh - thresh to make cloud mask out of cloud top temperature rather than cloud mask product
            sensor_zenith_max: None or degrees. Mark all data outside this as nan.
            satellite - 'GOES WEST'
                        'MSG+0415'
        Output:
            List of Scene objects
                Special attributes for Scenes:
                    self.static_file_filename - filename for static info like lon, lat, zenith, etc
                    self.pixel_area_size - from static file
                    self.nadir_lon
    """
    assert type(dates) == str or type(dates) == list, 'dates must be str or list'
    if type(dates) == str: dates = [dates]
    # if nearest_time is None and satellite=='GOES WEST': nearest_time = '2100'
    # elif nearest_time is None and satellite=='GOES EAST': nearest_time = '1700'
    # elif nearest_time is None and satellite=='MSG+0415': nearest_time = '0900'
    # elif nearest_time is None and satellite=='MSG+0000': nearest_time = '1200'
    # # elif nearest_time is None and satellite=='HIMAWARI+1407': nearest_time = '1700'
    # elif nearest_time is None and satellite=='HIMAWARI+1407': nearest_time = '0300'
    # elif nearest_time is None: raise ValueError('what time here?')
    # elif len(nearest_time) != 4: raise ValueError('Unsupported nearest time format "{}", should be %H%M'.format(nearest_time))

    if satellite == 'GOES WEST': 
        dir = dirs.goes_west_dir
        static_dir = dirs.goes_west_static_dir
        static_filenames = sorted(glob.glob(static_dir+'*.hdf'))
        noon = '2100'
        midn = '0900'
        resolution = 2
    elif satellite == 'GOES EAST': 
        dir = dirs.goes_east_dir
        static_dir = dirs.goes_east_static_dir
        static_filenames = sorted(glob.glob(static_dir+'*2km.hdf'))
        noon = '1700'
        midn = '0500'
        resolution = 2
    elif satellite == 'MSG+0415':
        dir = dirs.msg_0415_dir
        static_dir = dirs.msg_0415_static_dir
        static_filenames = sorted(glob.glob(static_dir+'*3km.hdf'))
        noon = '0900'
        midn = '2100'
        resolution = 3
    elif satellite == 'MSG+0000':
        dir = dirs.msg_0000_dir
        static_dir = dirs.msg_0000_static_dir
        static_filenames = sorted(glob.glob(static_dir+'*3km.hdf'))
        noon = '1200'
        midn = '0000'
        resolution = 3
    elif satellite == 'HIMAWARI+1407':
        dir = dirs.himawari_1407_dir
        static_dir = dirs.himawari_1407_static_dir
        static_filenames = sorted(glob.glob(static_dir+'*2km.hdf'))
        noon = '0300'
        midn = '1500'
        resolution = 2
    else: raise ValueError(f'Satellite {satellite} not supported')

    all_scenes = []
    if nearest_time == 'day': hours = [noon]
    elif nearest_time == 'night': hours = [midn]
    elif nearest_time is not None: hours = [nearest_time]
    else: hours = [noon, midn]

    for date in dates:
        folder_name = datetime.datetime.strptime(date, "%Y%j").strftime("%Y/%Y_%m_%d")
            
        date_times = [datetime.datetime.strptime(date+hour, '%Y%j%H%M') for hour in hours]
        
        for d in date_times:

            filenames = sorted(glob.glob(dir+folder_name+'/*CMA*{}*.nc'.format(d.strftime("%Y%m%dT%H%M"))))
            
            if filenames == []:
                print(f'No file for date {d}')
                continue
            elif len(filenames)> 1: raise ValueError('More than one file found for sat {} time '.format(satellite)+d.strftime("%Y%m%dT%H%M"))
            if static_filenames == []: raise ValueError('Static file not found for sat {} time {}'.format(satellite, d.strftime("%Y%m%dT%H%M")))
            if len(static_filenames)> 1: raise ValueError('More than one static file found for sat {} time {}'.format(satellite, d.strftime("%Y%m%d")))

            new_scene = Scene(None, satellite)
            new_scene.file_filename = filenames[0]
            new_scene.static_file_filename = static_filenames[0]
            new_scene.altitude = 35785 # km
            new_scene.nadir_resolution = resolution # km
            new_scene.sensor_zenith_max = sensor_zenith_max
            new_scene.cloud_temp_thresh = cloud_temp_thresh
            new_scene.date = d
            new_scene.land_sea_truncation = land_sea
            new_scene._filename = '{}_{}_Scene_{}_{}_{}'.format('GEOSTA ',satellite, sensor_zenith_max, d, nearest_time).replace(' ','_').replace('/','_')+'/'
            new_scene.lat_range = lat_range
            all_scenes.append(new_scene)

    return all_scenes
def POLDER_filenames(dates, 
                    cloud_thresh=50,
                    n_dir=None, optical_depth_thresh=None, 
                    sensor_zenith_max=50, lat_range=None, land_sea=None):
    """
        Input:
            dates - Dates to look for data: list or str, format yyyyddd
            cloud_thresh - Threshold to consider cloudy. Clouds are >= this: int, percent
            optical_depth_thresh - If other than None, will use an optical depth thresh instead of cloud mask
            n_dir - Number of direction to use for each pixel. Default is first
            sensor_zenith_max: None or degrees. Mark all data outside this as nan.
        Output:
            List of Scene objects
                Special attributes for Scenes:
                    self.resolution_degrees = resolution_degrees
    """
    assert type(dates) == str or type(dates) == list, 'dates must be str or list'
    if type(dates) == str: dates = [dates]
    all_scenes = []
    if sensor_zenith_max != 50:
        warn('Sensor Zenith Max is {}, not the default of 50. Values larger than 50 are dangerous; see journal 6/20/22')

    for date in dates:
        folder_name = datetime.datetime.strptime(date, "%Y%j").strftime("%Y/%Y_%m_%d")
        filenames = sorted(glob.glob(dirs.polder_dir + folder_name + '/PARASOL_PM02-L2*.hdf'))
        for i, filename in enumerate(filenames):
            new_scene = Scene(filename, 'POLDER')
            new_scene.altitude = 705 # km
            new_scene.resolution_degrees = 1/18 # km
            new_scene.nadir_resolution = new_scene.resolution_degrees * 111 # km
            new_scene.sensor_zenith_max = sensor_zenith_max
            new_scene._filename = '{}_Scene_{}_{}_{}_{}_{}_{}/'.format('POLDER', sensor_zenith_max, date, optical_depth_thresh, n_dir, cloud_thresh, i)
            new_scene.land_sea_truncation = land_sea
            new_scene.lat_range = lat_range
            all_scenes.append(new_scene)

    return all_scenes
def VIIRS_filenames(dates, n_files_to_combine=24, time=None, sensor_zenith_max = 60, cloud_thresh=0, suppress_missing_warning=True, lat_range=None, land_sea=None):
    """
        Input:
            dates - Dates to look for data: list or str, format yyyyddd
            cloud_thresh -      inclusive
            n_files_to_combine - Combine this many files into one Scene: int 
                                    must be multiple of 288 (there are 288 files per day)
            time - If fewer than all files for each day are desired, specify the time(s) desired: str or list
            suppress_missing_warning - when file is missing, print info about it: bool
            sensor_zenith_max: None or degrees. Mark all data outside this as nan.
            cloud_thresh: int: 0,1,2,3 consider values <= this as cloudy. 0= conf. cloudy, 3=conf. clear, 
        Output:
            List of Scene objects
                Special attributes for Scenes:
                    - n_files_to_combine
                    - optical_depth_thresh
    """
    if type(time) == str: time = [time]
    if type(dates) == str: dates = [dates]
    assert 240/n_files_to_combine == 240//n_files_to_combine, "n_files_to_combine must be multiple of 240"
    all_scenes = []
    if time is None:
        all_hours = []
        for hour in range(24):
            for minute in range(0,60,6):
                all_hours.append(datetime.time(hour=hour, minute=minute).strftime('%H%M'))
    else:
        if len(time) == 1 and n_files_to_combine > 1: n_files_to_combine = 1
        all_hours = time


    for date in dates:
        folder_name = datetime.datetime.strptime(date, "%Y%j").strftime("%Y/%j/")
        for n_chunk in range(240 // n_files_to_combine):
            if n_chunk >= len(all_hours):
                break
            times = range(n_chunk * n_files_to_combine, (n_chunk + 1) * n_files_to_combine)
            filenames = []
            omit = False
            for hour in times:
                time = all_hours[hour]
                # Need to read both level 3 and level 6
                vir_file = glob.glob(dirs.viirs_dir + folder_name + 'CLDMSK_L2_VIIRS_SNPP.A' + date + '.' + time + '*.nc')
                if len(vir_file) == 0:
                    if not suppress_missing_warning:
                        print("Missing file at "+folder_name+' '+time+', omitting this orbit')
                    omit = True
                    break
                assert len(vir_file) == 1, "More than one file available for "+folder_name+' '+time
                filenames.append(vir_file[0])
            if not omit:
                new_scene = Scene(filenames, 'VIIRS')
                new_scene.n_files_to_combine = n_files_to_combine
                new_scene.altitude = 829 # km
                new_scene.nadir_resolution = 0.77 #km. From metadata: VIIRS/NOAA20 Cloud Properties 6-min L2 Swath 750m, from latlon calculations its 0.77
                new_scene.sensor_zenith_max = sensor_zenith_max
                new_scene.date = datetime.datetime.strptime(date, "%Y%j")
                new_scene.lat_range = lat_range
                new_scene.land_sea_truncation = land_sea
                new_scene.cloud_thresh = cloud_thresh
                new_scene._filename = '{}_Scene_{}_{}_{}'.format('VIIRS', sensor_zenith_max, date, times, n_files_to_combine).replace(' ','_').replace('/','_')+'/'
                all_scenes.append(new_scene)
    return all_scenes
def EPIC_filenames(dates, optical_depth_thresh=None, one_per_day=True, sensor_zenith_max = 60, cloud_thresh=4, lat_range=None, land_sea=None):
    """
        Input:
            dates - Dates to look for data: list or str, format yyyyddd
            optical_depth_thresh - If other than None, will use an optical depth thresh instead of cloud mask
            one_per_day - Return only the noon image: bool
            sensor_zenith_max: None or degrees. Mark all data outside this as nan.
            cloud_thresh: int: 1,2,3,4 consider values >= this as cloudy. 1= conf.clear, 
        Output:
            List of Scene objects
                Special attributes for Scenes:
                    -cloud_thresh
                    -optical_depth_thresh
    """
    if type(dates) == str: dates = [dates]

    all_scenes = []

    for date in dates:
        file_time_name = datetime.datetime.strptime(date, "%Y%j").strftime("%Y%m%d")
        filenames = sorted(glob.glob(dirs.epic_dir+date[:4]+'_V3/DSCOVR_EPIC_L2_CLOUD*'+file_time_name + '*.nc4'))
        if filenames == []:
            continue
        if one_per_day:
            filenames = [filenames[0]]    # first for day
        elif one_per_day == 'omit-glint':
            # do not include 0000Z-1000Z, due to sun glint, as karlie does
            filenames = [f for f in filenames if int(f[-13:-9])>1000]
        for i, filename in enumerate(filenames): 
            date_time_object = datetime.datetime.strptime(filename[-21:-9], "%Y%m%d%H%M")  

            new_scene = Scene(filename, 'EPIC')
            new_scene.altitude = 1.5e6 # km
            new_scene.nadir_resolution = 8 #km
            new_scene.sensor_zenith_max = sensor_zenith_max
            new_scene.optical_depth_thresh = optical_depth_thresh
            new_scene.cloud_thresh = cloud_thresh
            new_scene.date = date_time_object
            new_scene.lat_range = lat_range
            new_scene.land_sea_truncation = land_sea
            new_scene._filename = '{}_Scene_{}_{}_{}_{}_{}'.format('EPIC', sensor_zenith_max, date, optical_depth_thresh, cloud_thresh,i).replace(' ','_').replace('/','_')+'/'
            all_scenes.append(new_scene)

    return all_scenes
def GEORING_filenames(dates, one_per_day=False, lat_range=(-60,60), resolution_degrees=0.1, nearest_time=None, sensor_zenith_max=None, land_sea=None, cloud_temp_thresh = None):
    """
        Input:
            dates - Dates to look for data: list or str, format yyyyddd
            nearest_time - If a specific time is desired. Must set one_per_day=True: str
            one_per_day - Return only the noon image: bool
            sensor_zenith_max: None or degrees. Mark all data outside this as nan.
            cloud_temp_thresh: if None, use cloud mask, otherwise create one using clouds colder than this value.
        Output:
            List of Scene objects
                Special attributes for Scenes:
                    self.resolution_degrees = resolution_degrees
    """
    if type(dates) == str: dates = [dates]
    if nearest_time is None: nearest_time = '0000'
    if lat_range is None: raise ValueError('GEORING should be limited between -60 and 60')
    if sensor_zenith_max is not None: raise ValueError('GEORING should not be limited by sensor zenith')

    all_scenes = []

    for date in dates:
        if one_per_day:
            file_time_names = [datetime.datetime.strptime(date+nearest_time, '%Y%j%H%M')]
        else:
            hours = ['0000', '1200']
            file_time_names = [datetime.datetime.strptime(date+hour, '%Y%j%H%M') for hour in hours]
        for d in file_time_names:

            if cloud_temp_thresh is None:
                filenames = sorted(glob.glob(dirs.georing_dir+date[:4]+'/*CMA_PLATECARREE*{}*.nc'.format(d.strftime("%Y%m%dT%H%M"))))
            else:
                filenames = sorted(glob.glob(dirs.georing_dir+date[:4]+'/*CTTH_PLATECARREE*{}*.nc'.format(d.strftime("%Y%m%dT%H%M"))))
            if filenames == []:
                continue
            elif len(filenames)> 1: 
                print(filenames)
                raise ValueError('More than one file found for time '+d.strftime("%Y%m%dT%H%M"))


            new_scene = Scene(filenames[0], 'GEORING')
            new_scene.altitude = 1.5e6 # km
            new_scene.nadir_resolution = resolution_degrees*111 #km
            new_scene.resolution_degrees = resolution_degrees
            new_scene.sensor_zenith_max = sensor_zenith_max
            new_scene.date = d
            new_scene.lat_range = lat_range
            new_scene.land_sea_truncation = land_sea
            new_scene.cloud_temp_thresh = cloud_temp_thresh
            new_scene._filename = '{}_Scene_{}_{}_{}_{}'.format('GEORING', lat_range, resolution_degrees, sensor_zenith_max, date).replace(' ','_').replace('/','_')+'/'
            all_scenes.append(new_scene)

    return all_scenes
def SAM_filenames(timestamps, run='TWPICE.RADSNOW.800m',threshold=1, slices = False):
    """
        slices: T/F if True, create scene for each slice. If false, smash the 3D field
        threshold: for slices=True, it is g/kg per grid box (probably use .01)
                    for slices=False, it is cloud liquid water path + ice water path, probably use 1 (mm)

            For TWPICE.RADSNOW.800m, 
                timesteps go from 000060 to 293760, every 60 seconds, for 4096 total.
                                First clouds are at timestep 25 (starting with first at 0)
                threshold is applied to a vertical sum of cloud water and cloud ice (units mm)
            For TWPICE.100m, 
                timesteps go from 0000000150 to 0000003450 every 150 (s)
                threshold is applied to a vertical sum of cloud water and cloud ice (units mm)
            For GATE_IDEAL, 
                timesteps go from to 0000001800 to 0000043200 every 150 (s). "steady state" is from 0000021600 on.
    """
    if slices and threshold >= 1: warn('Using threshold of {} g/kg for each grid box which is very high'.format(threshold))
    if slices != 1 and slices != 0: raise ValueError('slices must be T/F')

    if run == 'TWPICE.RADSNOW.800m':
        all_timestamps = [str(i)[1:] for i in range(1000060, 1293820, 60)]
        n_levels = 255
    elif run == 'TWPICE.100m':
        all_timestamps = [str(i)[1:] for i in range(10000000150, 10000003600, 150)]
        n_levels = 255
    elif run == 'RCE295':
        all_timestamps = [str(i)[1:] for i in range(10000002880, 10000172800, 2880)]
        n_levels = 80
    elif run == 'GATE_IDEAL':
        all_timestamps = [str(i)[1:] for i in range(10000001800, 10000043200+1800, 1800)]   
        # all_timestamps = [str(i)[1:] for i in range(10000021600, 10000043350, 1800)]
        n_levels = 210
    else: raise ValueError('Run {} not supported'.format(run))

    if timestamps == 'all': 
        timestamps = all_timestamps

    all_scenes = []

    for timestamp in timestamps:

        if run == 'TWPICE.RADSNOW.800m':
            filename = '{}TWPICE_800m_256L_RADSNOW_64_0000{}.nc.gz'.format(dirs.twpice_800m, timestamp)
            res = .8 # km
        elif run == 'TWPICE.100m':
            filename = ('{}OUT_3D.QC/TWPICE_LPT_3D_QC_{}.nc'.format(dirs.twpice_100m, timestamp),
                        '{}OUT_3D.QI/TWPICE_LPT_3D_QI_{}.nc'.format(dirs.twpice_100m, timestamp),
                        '{}OUT_3D.QS/TWPICE_LPT_3D_QS_{}.nc'.format(dirs.twpice_100m, timestamp),)
            res = .1
        elif run == 'RCE295':
            filename = dirs.rce295+'RCE_400x400x95_295K_FR_{}.nc'.format(timestamp)
            res = .5
        elif run == 'GATE_IDEAL':
            filename = (dirs.gate_ideal+'GATE_IDEAL_2048x2048_qn_{}.dat'.format(timestamp),
                        dirs.gate_ideal+'GATE_IDEAL_2048x2048_qp_{}.dat'.format(timestamp),
                        dirs.gate_ideal+'GATE_IDEAL_2048x2048_tab_{}.dat'.format(timestamp))
            res = .1

        if slices: 
            scenes = []
            for i in range(n_levels):
                new_scene = SamScene(filename, run, threshold, slice=i)
                new_scene.nadir_resolution = res
                new_scene.timestamp = timestamp
                scenes.append(new_scene)
            all_scenes.extend(scenes)

        else:
            scene = SamScene(filename, run, threshold)
            scene.nadir_resolution = res
            scene.timestamp = timestamp
            scene.slices = False
            all_scenes.append(scene)

    return all_scenes

def _MODIS_5km_read(scene):
    """
        Input: 
            scene - Scene object: Scene
            ask_to_delete - When a corrupted file is found, ask whether it should be deleted
        Output:
            scene - Scene object with attributes assigned
                Special attributes for Scenes:
                    - n_files_to_combine
                    - cloud_thresh

        Assign data attributes to Scenes

    """

    # Load first files:
    try:
        MOD06_file = SD(scene.file_filename[0], SDC.READ)
    except HDF4Error:
        print(f'Failed to read file, returning empty scene: \n       {scene.file_filename[0]}\n    Error msg: {e}')
        scene = empty_scene(scene)
        return scene

    lat = MOD06_file.select('Latitude')[:]
    lon = MOD06_file.select('Longitude')[:]
    cloud_fraction = MOD06_file.select('Cloud_Fraction')[:]  # in percent
    cloud_mask_flags = MOD06_file.select('Cloud_Mask_5km')[:]
    sensor_zenith = MOD06_file.select('Sensor_Zenith')[:] * 0.009999999776482582  # scale factor to put in deg
    sensor_azimuth = MOD06_file.select('Sensor_Azimuth')[:] * 0.009999999776482582  # scale factor to put in deg
    date_seconds = MOD06_file.select('Scan_Start_Time')[:].astype('timedelta64[s]')
    del MOD06_file
    # append the rest:
    for file in scene.file_filename[1:]:
        try:
            new_MOD06_file = SD(file, SDC.READ)
        except HDF4Error as e:
            print(f'Failed to read file, returning empty scene: \n       {file}\n    Error msg: {e}')
            scene = empty_scene(scene)
            return scene
        new_lat = new_MOD06_file.select('Latitude')[:]
        new_lon = new_MOD06_file.select('Longitude')[:]
        new_cloud_fraction = new_MOD06_file.select('Cloud_Fraction')[:]  # in percent
        new_cloud_mask_flags = new_MOD06_file.select('Cloud_Mask_5km')[:]
        new_sensor_zenith = new_MOD06_file.select('Sensor_Zenith')[:] * 0.009999999776482582  # scale factor to put in deg
        new_sensor_azimuth = new_MOD06_file.select('Sensor_Azimuth')[:] * 0.009999999776482582  # scale factor to put in deg
        new_date_seconds = new_MOD06_file.select('Scan_Start_Time')[:].astype('timedelta64[s]')
        del new_MOD06_file

        lat = np.append(lat, new_lat, axis=0)
        lon = np.append(lon, new_lon, axis=0)
        cloud_fraction = np.append(cloud_fraction, new_cloud_fraction, axis=0)
        cloud_mask_flags = np.append(cloud_mask_flags, new_cloud_mask_flags, axis=0)
        sensor_zenith = np.append(sensor_zenith, new_sensor_zenith, axis=0)
        sensor_azimuth = np.append(sensor_azimuth, new_sensor_azimuth, axis=0)
        date_seconds = np.append(date_seconds, new_date_seconds, axis=0)
        del new_lat, new_lon, new_cloud_fraction, new_cloud_mask_flags, new_sensor_zenith, new_date_seconds

    cloud_mask = np.zeros(cloud_fraction.shape, dtype=np.float16)
    cloud_mask[cloud_fraction>=scene.cloud_thresh] = 1
    cloud_mask[cloud_fraction == 127] = np.nan  # according to metadata '_FillValue': 127


    lat[lat<=-999] = np.nan
    lon[lon<=-999] = np.nan  # according to metadata '_FillValue': -999.9000244140625
    sensor_zenith[sensor_zenith<-300] = np.nan   # according to metadata '_FillValue': -32768 for zenith and azimuth
    sensor_azimuth[sensor_azimuth<-300] = np.nan # which is -327.67999267578125 with the scale factor. but they should be between -180 and 180

    cloud_mask_flags = cloud_mask_flags[:,:,0]
    format_vectorized = np.vectorize(format)
    cloud_mask_flags = format_vectorized(cloud_mask_flags, '08b')
    day_night = MODIS_VIIRS_cloudmask_flag(cloud_mask_flags, 'day_night')

    date = np.full(cloud_fraction.shape, np.datetime64('1993-01-01 00:00:00'))
    date = date+date_seconds

    # Set attributes:
    scene.cloud_mask = uf.encase_in_value(cloud_mask)
    scene.lon = uf.encase_in_value(lon)
    scene.lat = uf.encase_in_value(lat)
    scene.day_night = uf.encase_in_value(day_night)
    scene.sensor_zenith = uf.encase_in_value(sensor_zenith)
    scene.sensor_azimuth = uf.encase_in_value(sensor_azimuth)
    scene.date = uf.encase_in_value(date, dtype=date.dtype, value=np.datetime64("NaT"))
    scene.sun_glint = []
    scene.optical_depth = []
    scene.shape = scene.cloud_mask.shape

    return scene

def _MODIS_1km_read(scene):
    """
        Input: 
            scene - Scene object: Scene
            ask_to_delete - When a corrupted file is found, ask whether it should be deleted
        Output:
            scene - Scene object with attributes assigned
                Special attributes for Scenes:
                    - n_files_to_combine

        Assign data attributes to Scenes

    """


    # Load first files:
    try:
        MOD03_file = SD(scene.file_filename[0][0], SDC.READ)
    except HDF4Error:
        print(f'Failed to read file, returning empty scene: \n       {scene.file_filename[0][0]}\n    Error msg: {e}')
        scene = empty_scene(scene)
        return scene
    try:
        MOD06_file = SD(scene.file_filename[0][1], SDC.READ)
    except HDF4Error:
        print(f'Failed to read file, returning empty scene: \n       {scene.file_filename[0][1]}\n    Error msg: {e}')
        scene = empty_scene(scene)
        return scene

    lat = MOD03_file.select('Latitude')[:]
    lon = MOD03_file.select('Longitude')[:]
    cloud_mask_flags = MOD06_file.select('Cloud_Mask_1km')[:]
    if scene.tau_thresh is not None: 
        tau = MOD06_file.select('Cloud_Top_Pressure')[:] * .01    # scale factor
        tau_quality = MOD06_file.select('Quality_Assurance_1km')[:] 
    sensor_zenith = MOD03_file.select('SensorZenith')[:] * 0.01  # scale factor to put in deg
    sensor_azimuth = MOD03_file.select('SensorAzimuth')[:] * 0.01  # scale factor to put in deg
    del MOD03_file, MOD06_file
    # append the rest:
    for file in scene.file_filename[1:]:
        try:
            new_MOD03_file = SD(file[0], SDC.READ)
        except HDF4Error as e:
            print(f'Failed to read file, returning empty scene: \n       {file[0]}\n    Error msg: {e}')
            return scene
        try:
            new_MOD06_file = SD(file[1], SDC.READ)
        except HDF4Error:
            print(f'Failed to read file, returning empty scene: \n       {file[1]}\n    Error msg: {e}')
            scene = empty_scene(scene)
            return scene
        new_lat = new_MOD03_file.select('Latitude')[:]
        new_lon = new_MOD03_file.select('Longitude')[:]
        new_cloud_mask_flags = new_MOD06_file.select('Cloud_Mask_1km')[:]
        if scene.tau_thresh is not None: 
            new_tau = new_MOD06_file.select('Cloud_Optical_Thickness')[:] * .01    # scale factor
            new_tau_quality = MOD06_file.select('Quality_Assurance_1km')[:] 
            # new_tau = new_MOD06_file.select('Quality_Assurance_1km')[:]
        new_sensor_zenith = new_MOD03_file.select('SensorZenith')[:] * 0.01  # scale factor to put in deg
        new_sensor_azimuth = new_MOD03_file.select('SensorAzimuth')[:] * 0.01  # scale factor to put in deg
        del new_MOD03_file, new_MOD06_file

        lat = np.append(lat, new_lat, axis=0)
        lon = np.append(lon, new_lon, axis=0)
        cloud_mask_flags = np.append(cloud_mask_flags, new_cloud_mask_flags, axis=0)
        if scene.tau_thresh is not None: 
            tau = np.append(tau, new_tau, axis=0)
            tau_quality = np.append(tau_quality, new_tau_quality, axis=0)
        sensor_zenith = np.append(sensor_zenith, new_sensor_zenith, axis=0)
        sensor_azimuth = np.append(sensor_azimuth, new_sensor_azimuth, axis=0)
        # date_seconds = np.append(date_seconds, new_date_seconds, axis=0)
        del new_cloud_mask_flags, new_sensor_zenith, new_lat, new_lon, new_sensor_azimuth

    # Apply nans:

    sensor_zenith[sensor_zenith<-300] = np.nan   # according to metadata '_FillValue': -32767 for zenith and azimuth
    sensor_azimuth[sensor_azimuth<-300] = np.nan # which is -327.67999267578125 with the scale factor. but they should be between -180 and 180

    cloud_mask_flags = cloud_mask_flags[:,:,0]
    format_vectorized = np.vectorize(format)
    cloud_mask_flags = format_vectorized(cloud_mask_flags, '08b')
    cloud_mask = MODIS_VIIRS_cloudmask_flag(cloud_mask_flags, 'cloud_mask_value')
    day_night = MODIS_VIIRS_cloudmask_flag(cloud_mask_flags, 'day_night')

    if scene.tau_thresh is not None:
        
        tau_flags = format_vectorized(tau_quality[:,:,3], '08b')
        tau_nans = MODIS_tau_flag(tau_flags)
        tau_cloud_mask = np.zeros_like(tau, dtype=np.float32)
        tau_cloud_mask[tau > scene.tau_thresh] = 1 
        tau_cloud_mask[tau_nans==1] = np.nan
        tau_cloud_mask[np.isnan(cloud_mask)] = np.nan
        tau_cloud_mask[cloud_mask==0] = 0       # tau may be nan simply because pixel is clear
        cloud_mask = tau_cloud_mask
        del tau, tau_flags, tau_nans, tau_quality
    

    lat[lat<=-500] = np.nan
    lon[lon<=-500] = np.nan  # according to metadata '_FillValue': -999

    # Set attributes:
    scene.cloud_mask = uf.encase_in_value(cloud_mask)
    scene.lon = uf.encase_in_value(lon)
    scene.lat = uf.encase_in_value(lat)
    scene.day_night = uf.encase_in_value(day_night)
    scene.sensor_zenith = uf.encase_in_value(sensor_zenith)
    scene.sensor_azimuth = uf.encase_in_value(sensor_azimuth)
    scene.sun_glint = []
    scene.shape = scene.cloud_mask.shape

    return scene

def _MODIS_Homebrew_read(scene):

    mod02_fname, mod03_fname, mod06_fname, mod021_fname = scene.file_filename

    try:
        MOD02_file = SD(mod02_fname, SDC.READ)
    except HDF4Error as e:
        print(f'Failed to read file, returning empty scene: \n       {mod02_fname}\n    Error msg: {e}')
        scene = empty_scene(scene)
        return scene
    try:
        MOD06_file = SD(mod06_fname, SDC.READ)
    except HDF4Error as e:
        print(f'Failed to read file, returning empty scene: \n       {mod06_fname}\n    Error msg: {e}')
        scene = empty_scene(scene)
        return scene
    try:
        MOD03_file = SD(mod03_fname, SDC.READ)
    except HDF4Error as e:
        print(f'Failed to read file, returning empty scene: \n       {mod03_fname}\n    Error msg: {e}')
        scene = empty_scene(scene)
        return scene
    

    lat = MOD03_file.select('Latitude')[:]
    lon = MOD03_file.select('Longitude')[:]
    lat[lat<=-500] = np.nan
    lon[lon<=-500] = np.nan  # according to metadata '_FillValue': -999

    band_1 = MOD02_file.select('EV_250_RefSB')[0,:,:].astype(np.float32)
    # band_2 = MOD02_file.select('EV_250_RefSB')[1,:,:].astype(np.float32)
    # valid_range 0,32767
    band_1[band_1>60000] = np.nan
    # band_2[band_2>60000] = np.nan
    # Scale for reflectance
    band_1 = band_1 * 5.607e-5
    # band_2 = band_2 * 3.373e-5

    # Image
    if scene.include_image:
        try:
            MOD021_file = SD(mod021_fname, SDC.READ)
        except HDF4Error:
            print('Corrupted file :\n       ',mod021_fname)
            scene = empty_scene(scene)
            return scene
        red = MOD021_file.select('EV_250_Aggr1km_RefSB')[0,:,:].astype(np.float32)
        red[red>60000] = np.nan
        red = red*5.60759e-5
        blue = MOD021_file.select('EV_500_Aggr1km_RefSB')[0,:,:].astype(np.float32)
        blue[blue>60000] = np.nan
        blue = blue*5.963e-5
        green = MOD021_file.select('EV_500_Aggr1km_RefSB')[1,:,:].astype(np.float32)
        green[green>60000] = np.nan
        green = green*5.302e-5

        RGB = np.dstack([red, green, blue])
        RGB = RGB*2   # increase contrast

        scene.image = RGB

    cloud_mask = np.zeros_like(band_1)
    cloud_mask[band_1>scene.thresh] = 1
    cloud_mask[np.isnan(band_1)] = np.nan


    cloud_mask_flags = MOD06_file.select('Cloud_Mask_1km')[:]
    cloud_mask_flags = cloud_mask_flags[:,:,0]
    format_vectorized = np.vectorize(format)
    cloud_mask_flags = format_vectorized(cloud_mask_flags, '08b')
    sun_glint = MODIS_VIIRS_cloudmask_flag(cloud_mask_flags, 'sun_glint')

    sensor_zenith = MOD03_file.select('SensorZenith')[:] * 0.01  # scale factor to put in deg
    sensor_azimuth = MOD03_file.select('SensorAzimuth')[:] * 0.01  # scale factor to put in deg
    sensor_zenith[sensor_zenith<-300] = np.nan   # according to metadata '_FillValue': -32767 for zenith and azimuth
    sensor_azimuth[sensor_azimuth<-300] = np.nan # which is -327.67999267578125 with the scale factor. but they should be between -180 and 180

    # Make arrays same shape as cloud mask. This assumes changes in sensor zenith/azimuth/lat/lon are much smaller than pixel size (1km).
    sensor_zenith = np.repeat(np.repeat(sensor_zenith, 4,1), 4,0)
    sensor_azimuth = np.repeat(np.repeat(sensor_azimuth, 4,1), 4,0)
    lat = np.repeat(np.repeat(lat, 4,1), 4,0)
    lon = np.repeat(np.repeat(lon, 4,1), 4,0)
    sun_glint = np.repeat(np.repeat(sun_glint, 4,1), 4,0)


    scene.cloud_mask = uf.encase_in_value(cloud_mask)
    scene.sun_glint = uf.encase_in_value(sun_glint)
    scene.sensor_zenith = uf.encase_in_value(sensor_zenith)
    scene.sensor_azimuth = uf.encase_in_value(sensor_azimuth)
    scene.lon = uf.encase_in_value(lon)
    scene.lat = uf.encase_in_value(lat)
    scene.shape = scene.cloud_mask.shape

    return scene

def _GEOST_read(scene):
    """
        Input: 
            scene - Scene object: Scene
        Output:
            scene - Scene object with attributes assigned
                Special attributes for Scenes:
                    -target_time: Time we want the closest image to: datetime.Datetime
                    -optical_depth_thresh: If cloud mask based on optical depth. Otherwise None

        Assign data attributes to Scenes. For GOES, optical depth is at lower resolution 
        than cloud mask, so everything will be at this resolution for optical depth.

    """


    try:
        geo_file = Dataset(scene.file_filename,'r')
        static_file = SD(scene.static_file_filename, SDC.READ)
    except Exception as e:
        print(f'Failed to read file, returning empty scene: \n       {scene.file_filename}\n    Error msg: {e}')
        return scene

    geo_file.set_auto_mask(False)
    cloud_mask = geo_file['cma'][:].astype(np.float32)
    cloud_mask[cloud_mask==255] = np.nan

    if scene.cloud_temp_thresh is not None:
        temp_file = Dataset(scene.file_filename.replace('CMA','CTTH'),'r')
        temp_file.set_auto_mask(False)
        # data is automatically scaled
        temp = temp_file['ctth_tempe'][:].astype(np.float32)
        temp_quality = temp_file['ctth_quality'][:]
        temp[temp_quality==24] = np.nan # bad 
        temp[temp_quality==32] = np.nan # interpolated
        temp[temp>785] = np.nan # fill value is 785.35 after scaling

        temp[np.isnan(cloud_mask)] = np.nan

        temp_mask = np.zeros_like(temp)
        temp_mask[temp < scene.cloud_temp_thresh]  = 1
        temp_mask[np.isnan(temp)] = np.nan

        temp_mask[cloud_mask==0]  = 0 # simply cloud free

        cloud_mask = temp_mask


    lon = static_file.select('Longitude')[:]
    lat = static_file.select('Latitude')[:]
    lon[lon==-999] = np.nan
    lat[lat==-999] = np.nan

    sensor_zenith = static_file.select('View_Zenith')[:]
    sensor_azimuth = static_file.select('View_Azimuth')[:]
    pixel_size = static_file.select('Pixel_Area_Size')[:]
    sensor_azimuth[sensor_azimuth==-999] = np.nan
    sensor_zenith[sensor_zenith==-999] = np.nan
    pixel_size[pixel_size==-999] = np.nan

    if scene.satellite_name == 'GOES WEST': nadir_lon = -137.2
    elif scene.satellite_name == 'GOES EAST': nadir_lon = -75.2
    elif scene.satellite_name == 'MSG+0415' : nadir_lon = 41.5
    elif scene.satellite_name == 'MSG+0000' : nadir_lon = 0
    elif scene.satellite_name == 'HIMAWARI+1407' : nadir_lon = 140.7

    scene.cloud_mask = uf.encase_in_value(cloud_mask)
    scene.lon = uf.encase_in_value(lon)
    scene.lat = uf.encase_in_value(lat)
    scene.sensor_zenith = uf.encase_in_value(sensor_zenith)
    scene.sensor_azimuth = uf.encase_in_value(sensor_azimuth)
    scene.pixel_area_size = uf.encase_in_value(pixel_size)
    scene.nadir_lon = nadir_lon
    scene.day_night = []
    scene.sun_glint = []
    scene.optical_depth = []
    scene.shape = scene.cloud_mask.shape

    return scene

def _POLDER_read(scene):
    """
        Input: 
            scene - Scene object: Scene
        Output:
            scene - Scene object with attributes assigned
                Special attributes for Scenes:
                    self.resolution_degrees = resolution_degrees

        Assign data attributes to Scenes. 

    """

    date = datetime.datetime.strptime(scene.file_filename[-29:-10], '%Y-%m-%dT%H-%M-%S')
    try:
        pold_file = SD(scene.file_filename, SDC.READ)
    except HDF4Error as e:
        print(f'Failed to read file, returning empty scene: \n       {scene.file_filename}\n    Error msg: {e}')
        scene = empty_scene(scene)
        return scene


    lat = pold_file.select('Latitude')[:]
    lon = pold_file.select('Longitude')[:]
    cloud_mask_flags = pold_file.select('cloud_classification')[:]

    # Take solar zenith and sensor zenith from location where sensor zenith was a minimum
    ind = np.nanargmin(pold_file.select('p_vza')[:], axis = 0)
    ind = np.expand_dims(ind, axis=0)
    # mus = np.squeeze(np.take_along_axis(pold_file.select('p_sza')[:], ind, axis=0))
    sensor_zenith = np.squeeze(np.take_along_axis(pold_file.select('p_vza')[:], ind, axis=0)).astype(np.float32)
    del pold_file

    # day_night = np.zeros(mus.shape, dtype=np.float32)

    cloud_mask = np.zeros_like(cloud_mask_flags, dtype=np.float32)
    cloud_mask[np.logical_or(cloud_mask_flags==5, cloud_mask_flags==15)] = 1

    # Add nans
    cloud_mask[cloud_mask_flags==-1] = np.nan
    lat[lat==-999] = np.nan 
    lon[lon==-999] = np.nan 
    # day_night[mus==65535] = np.nan
    sensor_zenith[sensor_zenith==65535] = np.nan

    # Scale and add offset:
    # mus = mus * .002
    sensor_zenith = sensor_zenith * .002

    # day_night[mus<89 & np.isfinite(day_night)] = 1
    # print('Day: {}, Night: {}, Nan: {}'.format(np.count_nonzero(day_night==1),np.count_nonzero(day_night==0),np.count_nonzero(np.isnan(day_night))))
    # Solar zenith not looking at all right

    # Flip if orbit closer to 180° lon
    # if np.abs(np.nanmean(lon[np.logical_and(np.isfinite(lat), np.abs(lat)<45)])) > 90:
    #     cloud_mask = flip_POLDER_data(cloud_mask)
    #     lon = flip_POLDER_data(lon)
    #     lat = flip_POLDER_data(lat)
    #     sensor_zenith = flip_POLDER_data(sensor_zenith)
    #     sensor_azimuth = flip_POLDER_data(sensor_azimuth)
    #     day_night = flip_POLDER_data(day_night)
    #     nan_mask = flip_POLDER_data(nan_mask).astype(bool)



    scene.cloud_mask = uf.encase_in_value(cloud_mask)
    scene.lon = uf.encase_in_value(lon)
    scene.lat = uf.encase_in_value(lat)
    scene.sensor_zenith = uf.encase_in_value(sensor_zenith)
    # scene.day_night = uf.encase_in_value(day_night)
    scene.day_night = []
    scene.date = date
    scene.sun_glint = []
    scene.shape = scene.cloud_mask.shape

    return scene

def _VIIRS_read(scene):
    """
        Input: 
            scene - Scene object: Scene
            ask_to_delete - When a corrupted file is found, ask whether it should be deleted
        Output:
            scene - Scene object with attributes assigned
                Special attributes for Scenes:
                    -n_files_to_combine

        Assign data attributes to Scenes
    """

    # Load first files:
    try:
        vir_file = Dataset(scene.file_filename[0],'r')
    except Exception as e:
        print(f'Failed to read file, returning empty scene: \n       {scene.file_filename[0]}\n    Error msg: {e}')
        return scene


    vir_file['geophysical_data'].variables['Integer_Cloud_Mask'].set_auto_mask(False)  # do this myself
    cloud_mask_flags = vir_file['geophysical_data']['Integer_Cloud_Mask'][:]
    sensor_zenith = vir_file['geolocation_data']['sensor_zenith'][:]
    sensor_azimuth = vir_file['geolocation_data']['sensor_azimuth'][:]
    lat = vir_file['geolocation_data']['latitude'][:]
    lon = vir_file['geolocation_data']['longitude'][:]
    date = vir_file['scan_line_attributes']['scan_start_time']


    del vir_file
    # append the rest:
    for file in scene.file_filename[1:]:
        try:
            new_vir_file = Dataset(file,'r')
        except HDF4Error as e:
            print(f'Failed to read file, returning empty scene: \n       {file}\n    Error msg: {e}')
            # if delete == 'y':
            #     os.remove(scene.file_filename[0])
            scene = empty_scene(scene)
            return scene

        new_vir_file['geophysical_data'].variables['Integer_Cloud_Mask'].set_auto_mask(False)  # do this myself
        new_cloud_mask_flags = new_vir_file['geophysical_data']['Integer_Cloud_Mask'][:]
        new_sensor_zenith = new_vir_file['geolocation_data']['sensor_zenith'][:]
        new_sensor_azimuth = new_vir_file['geolocation_data']['sensor_azimuth'][:]
        new_lat = new_vir_file['geolocation_data']['latitude'][:]
        new_lon = new_vir_file['geolocation_data']['longitude'][:]
        del new_vir_file

        lat = np.append(lat, new_lat, axis=0)
        lon = np.append(lon, new_lon, axis=0)
        cloud_mask_flags = np.append(cloud_mask_flags, new_cloud_mask_flags, axis=0)
        sensor_zenith = np.append(sensor_zenith, new_sensor_zenith, axis=0)
        sensor_azimuth = np.append(sensor_azimuth, new_sensor_azimuth, axis=0)
        del new_lat, new_lon, new_cloud_mask_flags, new_sensor_zenith


    lat[lat<=-99] = np.nan     #_FillValue: -999.0
    lon[lon<=-200] = np.nan  # according to metadata '_FillValue': -999.0
    # For zenith and azimuth, "scale_factor: 0.01" but the current range is actually ~70-0 for zenith and ~180 to -180 for azimuth, so I will not use the scale factor and assume they are already in degrees.
    sensor_zenith[sensor_zenith<=-32700] = np.nan   # according to metadata '_FillValue': -32768 for zenith and azimuth
    # sensor_zenith = sensor_zenith *.01  # scale_factor: 0.01
    sensor_azimuth[sensor_azimuth<=-32700] = np.nan
    # sensor_azimuth = sensor_azimuth*.01
    

    cloud_mask = np.zeros_like(cloud_mask_flags, dtype=np.float32)
    cloud_mask[cloud_mask_flags<=scene.cloud_thresh] = 1
    cloud_mask[cloud_mask_flags==-1] = np.nan



    # Set attributes:
    scene.cloud_mask = uf.encase_in_value(cloud_mask)
    scene.lon = uf.encase_in_value(lon)
    scene.lat = uf.encase_in_value(lat)
    scene.day_night = []
    scene.sensor_zenith = uf.encase_in_value(sensor_zenith)
    scene.sensor_azimuth = uf.encase_in_value(sensor_azimuth)
    scene.optical_depth = []
    scene.date = date     # retain the simple granule time instantiated in viirs_filenames for saveability
    scene.sun_glint = []
    scene.optical_depth = []
    scene.shape = scene.cloud_mask.shape

    return scene

def _EPIC_read(scene):
    """
        Input: 
            scene - Scene object: Scene
            ask_to_delete - When a corrupted file is found, ask whether it should be deleted
        Output:
            scene - Scene object with attributes assigned
                Special attributes for Scenes:
                    -cloud_thresh

        Assign data attributes to Scenes
    """
    try:
        EPIC_file = Dataset(scene.file_filename,'r')
    except Exception as e:
        print(f'Failed to read file, returning empty scene: \n       {scene.file_filename}\n    Error msg: {e}')
        scene = empty_scene(scene)
        return scene

    EPIC_file['geophysical_data'].variables['Cloud_Mask'].set_auto_mask(False)  # do this myself
    cloud_mask = EPIC_file['geophysical_data']['Cloud_Mask'][:]
    sensor_zenith = EPIC_file['geolocation_data']['sensor_zenith'][:]
    sensor_azimuth = EPIC_file['geolocation_data']['sensor_azimuth'][:]
    lat = EPIC_file['geolocation_data']['latitude'][:]
    lon = EPIC_file['geolocation_data']['longitude'][:]

    # Make cloud mask binary, assign nans and scales:
    cloud_mask = EPIC_cloudmask_flag(cloud_mask, scene.cloud_thresh)
    # For zenith and azimuth, "scale_factor: 0.01" but the current range is actually ~70-0 for zenith and ~180 to -180 for azimuth, so I will not use the scale factor and assume they are already in degrees.
    sensor_zenith[sensor_zenith<=-32000] = np.nan
    sensor_azimuth[sensor_azimuth<=-32000] = np.nan
    sensor_azimuth[np.isnan(cloud_mask)] = np.nan   # for some reason, non-Earth pixels are not -32768 like metadata 
    lat[lat<-900] = np.nan  # _FillValue: -999.0
    lon[lon<-200] = np.nan  # _FillValue: -999.0

    # Set attributes:
    scene.cloud_mask = uf.encase_in_value(cloud_mask)
    scene.lon = uf.encase_in_value(lon)
    scene.lat = uf.encase_in_value(lat)
    scene.day_night = []
    scene.sensor_zenith = uf.encase_in_value(sensor_zenith)
    scene.sensor_azimuth = uf.encase_in_value(sensor_azimuth)
    scene.sun_glint = []
    scene.optical_depth = []
    scene.shape = scene.cloud_mask.shape
    return scene

def _GEORING_read(scene):
    """
        Input: 
            scene - Scene object: Scene
            ask_to_delete - When a corrupted file is found, ask whether it should be deleted
        Output:
            scene - Scene object with attributes assigned
                Special attributes for Scenes:
                    -n_files_to_combine
                    -optical_depth_thresh

        Assign data attributes to Scenes
    """

    try:
        geo_file = Dataset(scene.file_filename,'r')
    except Exception as e:
        print(f'Failed to read file, returning empty scene: \n       {scene.file_filename}\n    Error msg: {e}')
        scene = empty_scene(scene)
        return scene


    if scene.cloud_temp_thresh is None:
        geo_file['cma'].set_auto_mask(False)  # do this myself
        cloud_mask = geo_file['cma'][:].astype(np.float32)

        cloud_mask[cloud_mask==255] = np.nan
    else:
        geo_file['ctth'].set_auto_mask(False)  # do this myself
        temp = geo_file['ctth'][:].astype(np.float32)

        temp[temp==65535] = np.nan

        temp = temp*0.01 + 130

        cloud_mask = np.zeros_like(temp)
        cloud_mask[temp<scene.cloud_temp_thresh] = 1
        cloud_mask[np.isnan(temp)] = np.nan




    lat = geo_file['Latitude'][:]
    lon = geo_file['Longitude'][:]
    
    
    # Set attributes:
    scene.cloud_mask = uf.encase_in_value(cloud_mask)[:,1:-1]       # Add nans to top and bottom but not edges as they wrap
    scene.lon = uf.encase_in_value(lon)[:,1:-1]
    scene.lat = uf.encase_in_value(lat)[:,1:-1]
    scene.shape = scene.cloud_mask.shape

    return scene

def _SAM_read(scene):
    try:
        if scene.run == 'TWPICE.RADSNOW.800m':
            if scene.file_filename.endswith('.gz'):
                with gzip.open(scene.file_filename) as gz:
                    with Dataset('dummy', mode='r', memory=gz.read()) as nc:
                        # print(nc.variables)
                        condensate = np.squeeze(nc['QN'][:])
                        scene.date = nc['time'][:]
                        scene.date_info = str(nc['time'])
            else:
                with Dataset(scene.file_filename, mode='r') as nc:
                    # print(nc.variables)
                    condensate = np.squeeze(nc['QN'][:])
                    scene.date = nc['time'][:]
                    scene.date_info = str(nc['time'])

            condensate = np.moveaxis(condensate, 0, 2)
            condensate = np.moveaxis(condensate, 0, 1)  # originally (z,y,x) but we want (x,y,z)
            if scene.slice is not None: raise ValueError('Need to convert 800m condensate to g/m^3')

        elif scene.run == 'TWPICE.100m':
            with Dataset(scene.file_filename[0], mode='r') as nc:
                # print(nc.variables)
                # for v in nc.variables:
                #     print(nc[v])
                water = np.squeeze(nc['QC'][:])
                scene.date = nc['time'][:]
                scene.date_info = str(nc['time'])
            with Dataset(scene.file_filename[1], mode='r') as nc:
                # print(nc.variables)
                ice = np.squeeze(nc['QI'][:])

            if scene.slice is not None:
                water = water[:,:,scene.slice]
                ice = ice[:,:,scene.slice]
                cloud_mask = water+ice
            else:  

                with Dataset(scene.file_filename[2], mode='r') as nc:
                    # print(nc.variables)
                    snow = np.squeeze(nc['QS'][:])
                # condensate = water+ice
                # Convert condensate from g/kg to g/m^3 (and then mm/m^2)
                rho = get_var_profile('RHO', scene.date, dirs.twpice_100m+'OUT_STAT/TWPICE_2048x256_100m_2s.nc')[:-1]
                z = get_var_profile('z', None, dirs.twpice_100m+'OUT_STAT/TWPICE_2048x256_100m_2s.nc')[:-1]     # stat file has an extra level on top
                midpoints = (z[1:]+z[:-1])/2
                dz = np.diff(midpoints)
                dz = np.append(25, dz)
                dz = np.append(dz, 285)
                # integrand = condensate*rho * dz

                # Water paths in g/m^2
                iwp = np.sum(ice*rho*dz, axis=2)    # cloud ice water path
                lwp = np.sum(water*rho*dz, axis=2)  # cloud liquid water path
                swp = np.sum(snow*rho*dz, axis=2)  # snow water path

                # Convert to optical depth
                cloud_mask = SAM_optical_depth(iwp,lwp,swp)

        elif scene.run == 'RCE295':

            with Dataset(scene.file_filename, mode='r') as nc:
                # print(nc.variables)
                # for v in nc.variables:
                #     print(nc[v])
                # exit()
                water_and_ice = np.squeeze(nc['QN'][:])
                snow_and_rain = np.squeeze(nc['QP'][:])
                
                temp = np.squeeze(nc['TABS'][:])
                scene.date = nc['time'][:]
                scene.date_info = str(nc['time'])

            water_and_ice = water_and_ice.transpose((2,1,0))  # originally (z,y,x) but we want (x,y,z)
            snow_and_rain = snow_and_rain.transpose((2,1,0))  # originally (z,y,x) but we want (x,y,z)
            temp = temp.transpose((2,1,0))  # originally (z,y,x) but we want (x,y,z)

            if scene.slice is not None:
                cloud_mask = water_and_ice[:,:,scene.slice]
                # print(cloud_mask.shape)
                # print(cloud_mask.max())
                # print(cloud_mask.min())
                # print(cloud_mask.mean())
                # print((water_and_ice[:,:,60]>=.01).sum()/400/400)
                # exit()
            else:

                # Convert condensate from g/kg to g/m^3 (and then mm/m^2)
                rho = get_var_profile('RHO', scene.date, dirs.rce295_stat)
                z = get_var_profile('z', None, dirs.rce295_stat)
                # print(z.shape)
                midpoints = (z[1:]+z[:-1])/2
                dz = np.diff(midpoints)
                dz = np.append(50, dz)
                dz = np.append(dz, 500)
                # print(dz.shape)
                # exit()

                qc, qi, _, qs, _ = SAM1MOM_hydro_part(temp, water_and_ice, snow_and_rain)
                # Water paths in g/m^2
                iwp = np.sum(qi*rho*dz, axis=2)    # cloud ice water path
                lwp = np.sum(qc*rho*dz, axis=2)  # cloud liquid water path
                swp = np.sum(qs*rho*dz, axis=2)  # snow water path

                cloud_mask = SAM_optical_depth(iwp,lwp,swp)

        elif scene.run == 'GATE_IDEAL':
            qn = np.fromfile(scene.file_filename[0], dtype='>f').reshape((2048,2048,210), order='F').astype(np.float32)
            tabs = np.fromfile(scene.file_filename[2], dtype='>f').reshape((2048,2048,210), order='F').astype(np.float32)

            if scene.slice is not None:
                press = np.fromfile('/uufs/chpc.utah.edu/common/home/krueger-group8/krueger-group3/krueger_grp/pbogen/sam/GATE_IDEAL_2048/OUT_2D/p.dat', dtype='<f')[:210]

                qn = qn[:,:,scene.slice]
                press = press[scene.slice]
                tabs = tabs[:,:,scene.slice]

                qvsat = samQVSAT.qvsat(tabs, press)*1000        # output is in kg/kg
                cloud_mask = qn/qvsat
            else:
                qp = np.fromfile(scene.file_filename[1], dtype='>f').reshape((2048,2048,210), order='F').astype(np.float32)


                rho = np.fromfile('/uufs/chpc.utah.edu/common/home/krueger-group8/krueger-group3/krueger_grp/pbogen/sam/GATE_IDEAL_2048/OUT_2D/rho.dat', dtype='<f')[:210]
                z = np.fromfile('/uufs/chpc.utah.edu/common/home/krueger-group8/krueger-group3/krueger_grp/pbogen/sam/GATE_IDEAL_2048/OUT_2D/z.dat', dtype='<f')[:210]

                midpoints = (z[1:]+z[:-1])/2
                dz = np.diff(midpoints)
                dz = np.append(50, dz)
                dz = np.append(dz, 293)
                print(f'Calculating {scene.run} Optical Depth. Memory used: {Process().memory_info().rss / 1024 ** 3:.01f} GB')
                qc, qi, _, qs, _ = SAM1MOM_hydro_part(tabs, qn, qp)
                # Water paths in g/m^2
                iwp = np.sum(qi*rho*dz, axis=2)    # cloud ice water path
                lwp = np.sum(qc*rho*dz, axis=2)  # cloud liquid water path
                swp = np.sum(qs*rho*dz, axis=2)  # snow water path

                cloud_mask = SAM_optical_depth(iwp,lwp,swp)
                print(f'Finished {scene.run} Optical Depth')

        scene.cloud_mask = np.zeros_like(cloud_mask)
        scene.cloud_mask[cloud_mask>scene.threshold] = 1
        # if self.slice in [10,30,60]:
        #     plt.pcolormesh(self.cloud_mask, cmap=plotting_functions.cloud_colors)
        #     plt.colorbar()
        #     plotting_functions.savefig(f'cloud mask, {self.slice}, thresh {self.threshold}')
        #     print('done plotting')
        #     # plt.clf()
        #     plt.close()
        #     # exit()


    except FileNotFoundError:
        print(f'File not found, omitting: {scene.file_filename}')
        scene.cloud_mask = np.zeros((256,256))
    


    scene.pixel_sizes_x = np.ones_like(scene.cloud_mask)*scene.nadir_resolution
    scene.pixel_sizes_y = np.ones_like(scene.cloud_mask)*scene.nadir_resolution
    scene.shape = scene.cloud_mask.shape

    return scene

def _SAM_get_3d_cloudmask(scene, first_only = False):
    # first_only: isolate the first 2 vertical slices only
    try:
        if scene.run == 'GATE_IDEAL':
            qn = np.fromfile(scene.file_filename[0], dtype='>f').reshape((2048,2048,210), order='F').astype(np.float32)
            tabs = np.fromfile(scene.file_filename[2], dtype='>f').reshape((2048,2048,210), order='F').astype(np.float32)
            if first_only:
                qn = qn[:2]
                tabs = tabs[:2]

            press = np.fromfile('/uufs/chpc.utah.edu/common/home/krueger-group8/krueger-group3/krueger_grp/pbogen/sam/GATE_IDEAL_2048/OUT_2D/p.dat', dtype='<f')[:210]

            qvsat = samQVSAT.qvsat(tabs, press)*1000        # output is in kg/kg
            cloud_mask = qn/qvsat

        else: raise ValueError(f'Run not supported: {scene.run}')

        scene.cloud_mask = np.zeros_like(cloud_mask)
        scene.cloud_mask[cloud_mask>scene.threshold] = 1

    except FileNotFoundError:
        print(f'File not found, omitting: {scene.file_filename}')
        scene.cloud_mask = np.zeros((256,256,256))

    scene.pixel_sizes_x, scene.pixel_sizes_y, scene.pixel_sizes_z = _SAM_pixel_sizes(scene.run, cloud_mask.shape)
    scene.shape = scene.cloud_mask.shape

    return scene



def _SAM_pixel_sizes(run, shape):
    if run == 'GATE_IDEAL':
        z = np.fromfile('/uufs/chpc.utah.edu/common/home/krueger-group8/krueger-group3/krueger_grp/pbogen/sam/GATE_IDEAL_2048/OUT_2D/z.dat', dtype='<f')[:210]
        midpoints = (z[1:]+z[:-1])/2
        dz = np.diff(midpoints)
        dz = np.append(50, dz)
        dz = np.append(dz, 293)
        dz = dz/1000 # to km
        pixel_sizes_x = np.ones(shape)* .1
        pixel_sizes_y = np.ones(shape)* .1
        pixel_sizes_z = np.moveaxis(np.repeat([np.repeat([dz], shape[0], axis=0)], shape[1], axis=0), 1, 0)
    else: raise ValueError(f'Run not supported in pixel sizes: {run}')

    return pixel_sizes_x, pixel_sizes_y, pixel_sizes_z


def empty_scene(scene=None, sat=None):
    """
        If data failed to be read from file, set all attributes to [] so calculations 
        will not include data, but the scene can still move through the procedure
        without raising error
    """
    if scene is None: scene = Scene('',sat)
    to_set_empty = list(vars(scene).keys())
    to_set_empty.remove('file_filename')
    to_set_empty.remove('satellite_name')
    to_set_empty.remove('_is_complete')
    to_set_empty.remove('_filename')
    for attr in to_set_empty:
        setattr(scene, attr, np.array([]))

    scene.lat_range = None
    return scene
    

# All sat tools:
def truncate_image_by_sensor_zenith(cloud_mask, sensor_zenith, sensor_zenith_max):
    '''
        Input:
            cloud_mask: binary cloud mask: np.ndarray
            sensor_zenith: array of zeniths of same shape as cloud_mask: np.ndarray
            sensor_zenith_max: maximum value. For everywhere above, cloud_mask will be set to nan.
        Output:
            cloud_mask: cloud mask with values where sen z > max set to nan
            sensor_zenith: sensor zenith with values where sen z > max set to nan
    '''

    cloud_mask[sensor_zenith > sensor_zenith_max] = np.nan
    cloud_mask[~np.isfinite(sensor_zenith)] = np.nan

    return cloud_mask
def truncate_image_by_latitude(cloud_mask, lat, minlat, maxlat):
    '''
        Input:
            cloud_mask: binary cloud mask: np.ndarray
            lat: array of latitudes of same shape as cloud_mask: np.ndarray
            minlat, maxlat: floats.
        Output:
            cloud_mask: cloud mask with values where sen z > max set to nan
    '''

    cloud_mask[lat >= maxlat] = np.nan
    cloud_mask[lat < minlat] = np.nan
    cloud_mask[~np.isfinite(lat)] = np.nan

    return cloud_mask
def truncate_image_by_surface(cloud_mask, land_sea, keep_value):
    # keep_value = 0: Set land to nan 1: Set sea to nan
    cloud_mask[land_sea!=keep_value] = np.nan
    return cloud_mask
def get_pixel_sizes(nadir_size, sensor_zenith, sensor_azimuth, satellite_height, increase_pixels=1):
    """
        Input:
            - nadir_size: Size of pixel, in km, when sensor_zenith = 0
            - sensor_zenith: Angle to the vertical (for the pixel) in degrees: 2-D np.ndarray
            - sensor_azimuth: Angle to north (for the pixel) in degrees: 2-D np.ndarray
            - satellite_height: Height, im km, of satellite above nadir
            - increase_pixels: 0 to 1: see journal entry
        Output:
            - x_pixel_sizes, y_pixel_sizes: Sizes of respective sides of the pixels.
                                                Array of same shape as sensor_zenith: 2-D np.ndarray
        See Thomas DeWitt's journal 2/24/23 for derivation
    """
    sa = sensor_azimuth/degree_per_rad
    sz = sensor_zenith/degree_per_rad

    cos_z = np.cos(sz)

    sqrt = np.sqrt(2)*np.sqrt(r_Earth**2\
                              +4*r_Earth*satellite_height\
                              +2*satellite_height**2\
                              +np.cos(2*sz)*r_Earth**2)

    d = 0.5 * (-2*r_Earth*np.cos(sz)+sqrt)

    larger_size = nadir_size*(increase_pixels*d/satellite_height + (1-increase_pixels))

    # print(np.count_nonzero(larger_size<=0))
    # exit()
    
    pixel_sizes_x = ((larger_size/cos_z)-larger_size)*np.sin(np.abs(sa))**2 + larger_size
    pixel_sizes_y = ((larger_size/cos_z)-larger_size)*np.cos(np.abs(sa))**2 + larger_size
    # pixel_sizes_x = larger_size/2 * (np.cos(sa)+np.sin(sa)/np.cos(sz))
    # pixel_sizes_y = larger_size/2 * (np.sin(sa)+np.cos(sa)/np.cos(sz))

    return pixel_sizes_x, pixel_sizes_y
def get_gridded_pixel_sizes(resolution_degrees, lat):
    """
        Get pixel sizes for datasets that have been projected onto a grid of constant size in degrees using lat/lon. Approximate.
    """
    pixel_sizes_y = np.ones_like(lat)*111*resolution_degrees  
    pixel_sizes_x = np.ones_like(lat)*111*resolution_degrees * np.cos(lat*np.pi/180)   # 111 km per deg lat

    pixel_sizes_x[~np.isfinite(lat)] = np.nan
    pixel_sizes_y[~np.isfinite(lat)] = np.nan

    return pixel_sizes_x, pixel_sizes_y

# SAM tools
def get_var_profile(var,day, stat_path):        # from corey mod slightly
    # var is 'RHO' for ex
    stat = Dataset(stat_path)
    stat.set_auto_mask(False)

    if day is not None:
        times = stat['time'][:]
        t_idx = get_time_idx(times,day)

        var_profile = stat[var][t_idx,:]
    else: 
        var_profile = stat[var][:]
    return var_profile
def get_time_idx(times, day):         # from corey mod slightly
    # times = list of times from stat file
    # day is the time that we are looking for
    t_idx = np.argmin(np.abs(times - day))
    return t_idx
def SAM_optical_depth(iwp, lwp, swp):
    # ice, liquid, and snow water paths

    # for water LWP = 0.6292 * tau * re (g/m^2)     # from steve krueger
    re = 10
    fac = 1 / (0.6292 * re)
    tau_cloud_water = fac * lwp
    # for ice: IWP = 0.350 * tau * re, re = 30
    re = 30; # um
    fac = 1 / (0.350 * re)
    tau_cloud_ice = fac * iwp           
    # for snow: IWP = 0.350 * tau * re, re = 300
    re = 300; # um
    fac = 1 / (0.350 * re)
    tau_snow = fac * swp
    return tau_cloud_ice+tau_cloud_water+tau_snow
def SAM1MOM_hydro_part(T, qn=None, qp=None):
    '''
    (From Corey)
    Hydrometeor Partitioning from Kharioutdinov and Randall 2003
    T - Temperature array
    qn - Non-precipitating Condensate array
    qp - Precipitating Water array

    Returns qc, qi, qr, qs, qg hyrdometeor arrays based on input (always in that order)

    NOTE: T, qn and qp must be the same shape or follow numpy broadcasting rules
    '''

    if (qn is not None):

        T0n = 273.16
        T00n = 253.16
        omega_n = np.maximum(0,np.minimum(1,(T-T00n)/(T0n-T00n)))
        qc = omega_n * qn
        qi = (1-omega_n)*qn

    if (qp is not None):
        T0p = 283.16
        T00p = 268.16
        omega_p = np.maximum(0,np.minimum(1,(T-T00p)/(T0p-T00p)))

        T0g = 283.16
        T00g = 223.16
        omega_g = np.maximum(0,np.minimum(1,(T-T00g)/(T0g-T00g)))

        qr = omega_p*qp
        qs = (1-omega_p)*(1-omega_g)*qp
        qg = (1-omega_p)*omega_g*qp

    if (qn is not None) & (qp is not None):
        return qc, qi, qr, qs, qg
    elif (qn is not None) & (qp is None):
        return qc, qi
    elif (qn is None) & (qp is not None):
        return qr, qs, qg
    else:
        print('No hyrometeor array provided.')
def SAM_get_3d_variable(scene, variable, first_only = False):
    """
        Load 3-D variable from SAM.

        variable: 
            Options:        (https://home.chpc.utah.edu/~u0034822/gigaLES)

                1)  u: (X Wind Component); units: m/s
                2)  v: (Y Wind Component); units: m/s
                3)  w: (Z Wind Component); units: m/s
                4)  pp: (Pressure Perturbation); units: Pa
                5)  tab: (Absolute Temperature); units: K
                6)  qv: (Water Vapor); units: g/kg
                7)  qn: (Non-precipitating condensate (water+ice)), units: g/kg
                8)  qp: (Precipitating water (rain+snow)), units: g/kg
        first_only: isolate the first 2 vertical slices only

        returns: 
            array of shape (x,y,z) corresponding to the specified variable, or 3-D array of 0s if file is not found (sends warning in this case). 
    """
    # 
    try:
        if scene.run == 'GATE_IDEAL':
            var = np.fromfile(scene.file_filename[0].replace('qn',variable), dtype='>f').reshape((2048,2048,210), order='F').astype(np.float32)
            if first_only:
                var = var[:2]

        else: raise ValueError(f'Run not supported: {scene.run}')

    except FileNotFoundError:
        warn(f'File not found, omitting: {scene.file_filename}')
        var = np.zeros((256,256,256))

    return var
def SAM_get_2d_variable(scene, variable):
    """
        Load 2-D variable from SAM 2-D output.

        returns: 
            array of shape (x,y) corresponding to the specified variable, or 2-D array of 0s if file is not found (sends warning in this case). 
    """
    # 
    try:
        if scene.run == 'GATE_IDEAL':
            # print(dirs.gate_ideal+f'OUT_2D/*{scene.timestamp}.2Dcom_1.nc')
            # exit()
            nc = Dataset(dirs.gate_ideal_2D+f'GATE_IDEAL_S_2048x2048x256_100m_2s_2048_{scene.timestamp}.2Dcom_1.nc')
            if variable == 'list': 
                for a in nc.variables:
                    print(a)
                exit()
            var = nc[variable][:]
            

        else: raise ValueError(f'Run not supported: {scene.run}')

    except FileNotFoundError:
        warn(f'File not found, omitting: \n{dirs.gate_ideal_2D}GATE_IDEAL_S_2048x2048x256_100m_2s_2048_{scene.timestamp}.2Dcom_1.nc')
        
        var = np.zeros((256,256))

    return var
# Geostationary/full disk tools:
def geolocate_GOES(data):
    """
        Input:
            data: goes2go data object, from e.g. goes_nearesttime()
        Output:
            (lon, lat)
                2-D np.ndarray's of lon, lat for each pixel

        Determine lon, lat for GOES geostationaries from goes2go data.

        Adapted from Brian Blaylock's work:
        from https://github.com/blaylockbk/pyBKB_v3/blob/master/BB_GOES/mapping_GOES16_TrueColor.ipynb
    """

    # Satellite height
    sat_h = data['goes_imager_projection'].perspective_point_height

    # Satellite longitude
    sat_lon = data['goes_imager_projection'].longitude_of_projection_origin

    # Satellite sweep
    sat_sweep = data['goes_imager_projection'].sweep_angle_axis

    # Create a pyproj geostationary map object
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

    # The projection x and y coordinates equals the scanning angle (in radians) multiplied by the satellite height
    # See details here: https://proj4.org/operations/projections/geos.html?highlight=geostationary
    x = data['x'][:] * sat_h
    y = data['y'][:] * sat_h

    # Perform cartographic transformation. That is, convert image projection coordinates (x and y)
    # to latitude and longitude values.
    XX, YY = np.meshgrid(x, y)
    lon, lat = p(XX, YY, inverse=True)
    lon[~np.isfinite(lon)] = np.nan
    lat[~np.isfinite(lat)] = np.nan
    return lon, lat
def geostationary_zenith(nadir_longitude, lon, lat):
    """
        Input: 
            nadir_longitude: float, longitude of geostationary satellite, degrees
                    Possibilities:
                        GOES-WEST -137.2
                        GOES-EAST -75.2 
            lon:            2-D np.ndarray, longitudes of pixels
            lat:            2-D np.ndarray, longitudes of pixels
        Output:
            np.ndarray of same shape as lon, lat, of sensor zenith angles (degrees)

        Given the location of a geostationary satellite (so lat=0), calculate 
        sensor zenith angles, assuming a sphere. Output in degrees.

        See Thomas DeWitt's journal 12-14-21 for derivation
    """
    if nadir_longitude != -137.2: raise NotImplementedError('Does not work')
    lon[lon>nadir_longitude+180] = lon[lon>nadir_longitude+180]-360 # the switch to positives throws it off
    h = 35785   # height of geostationary

    θ = (nadir_longitude-lon)/degree_per_rad
    φ = lat/degree_per_rad
    cosξ = np.cos(φ)*np.cos(θ*np.cos(φ))
    ξ = np.arccos(cosξ)

    z = np.arctan(r_Earth*np.sin(ξ)/(h+(r_Earth*np.sin(ξ)))) + ξ

    return z *degree_per_rad
def geostationary_azimuth(nadir_longitude, lon, lat):
    """
        Input: 
            nadir_longitude: float, longitude of geostationary satellite, degrees
                    Possibilities:
                        GOES-WEST -137.2
                        GOES-EAST -75.2 
            lon:            2-D np.ndarray, longitudes of pixels
            lat:            2-D np.ndarray, longitudes of pixels
        Output:
            np.ndarray of same shape as lon, lat, of sensor azimuthal angles (degrees)

        Given the location of a geostationary satellite (so lat=0), calculate 
        sensor azimuthal angles, assuming a sphere. Output in degrees.
        Convention is North is 0, positive east of N, negative west of N

        See Thomas DeWitt's journal 12-14-21 for derivation
    """
    lat = lat/degree_per_rad
    c = (nadir_longitude-lon)/degree_per_rad

    azimuth = np.arctan(np.sin(c)*np.sin(lat)/(np.cos(c)-np.cos(lat)*np.cos(c)*np.cos(lat)))

    # some fudging
    azimuth[lat<0] = azimuth[lat<0] - np.pi 
    azimuth[azimuth>np.pi] = azimuth[azimuth>np.pi] - 2*np.pi
    azimuth[azimuth<-np.pi] = azimuth[azimuth<-np.pi] + 2*np.pi

    return azimuth*degree_per_rad*-1
def disk_border_mask(matrix):
    """
        Input:
            matrix - As long as it has nan's outside of the disk, it works: 2-D np.ndarray
        Output:
            border_mask - matrix of 0's inside the disk and 1's outside.
                            Same shape as matrix: 2-D np.ndarray
        Get region that is outside the disk from matrix that has nans
    """
    nanmask = np.isnan(matrix).astype(int)
    return (nanmask - uf.clear_border_adjacent(nanmask)).astype(bool)
def clear_disk_border(matrix):
    """
        Input:
            matrix - Cloud mask: 2-D np.ndarray, with nan's outside the disk
        Output:
            cleared - Same as matrix but with clouds touching the border removed
        
    """
    edge_mask = disk_border_mask(matrix)
    with_edge = matrix.copy()
    with_edge[edge_mask] = 1
    cleared = uf.clear_border_adjacent(with_edge).astype(float)
    cleared[edge_mask] = np.nan
    cleared[np.isnan(matrix)] = np.nan
    return cleared
def disk_edge_clouds_only(matrix):
    """
        Input:
            matrix - Cloud mask: 2-D np.ndarray, with nan's outside the disk
        Output:
            cleared - Same as matrix but with clouds NOT touching the border removed
    """
    return (matrix - clear_disk_border(matrix)).round() # fix floating point errors


# Satellite specific tools:
@njit(parallel=False)
def MODIS_VIIRS_cloudmask_flag(cloud_mask_flags, parameter):
    """
        Input: 
            cloud_mask_flags: 2-D np.ndarray: first bite, formatted in binary strings
            parameter:  Which data field to look at. Options:
                'day_night'
                'cloud_mask_determined'
                'cloud_mask_value'
                'sun_glint'

        Output: np.ndarray (2D) of shape cloud_mask_flags:
                    'day_night':    0 = Night  / 1 = Day / np.nan if error
                    'cloud_mask_determined'     1 = Determined / np.nan = not determined
                    'cloud_mask_value':     1 = Cloudy / 0 = Confident Clear, Probably Clear, or Uncertain 
                                            / np.nan if error / np.nan if undetermined
                    'land_sea':       1 = Land 0 = Sea
                    'sun_glint':        1 = Yes 0 = No


    For MOD06_L2 Cloud_Mask_1km or Cloud_Mask_5km
            From description in data:
            ---------------------------                                   
                Bit fields within each byte are numbered from the left:                            
                7, 6, 5, 4, 3, 2, 1, 0.                                                            
                The left-most bit (bit 7) is the most significant bit.                             
                The right-most bit (bit 0) is the least significant bit.                           
                                                                                                    
                First Byte                                                                         
                                                                                                    
                bit field       Description                             Key                        
                ---------       -----------                             ---                        
                                                                                                    
                0               Cloud Mask Flag                      0 = Not  determined           
                                                                    1 = Determined                
                                                                                                    
                2, 1            Unobstructed FOV Quality Flag        00 = Cloudy                   
                                                                    01 = Uncertain                
                                                                    10 = Probably  Clear          
                                                                    11 = Confident  Clear         
                                PROCESSING PATH                                                    
                                ---------------                                                    
                3               Day or Night Path                    0 = Night  / 1 = Day          
                4               Sunglint Path                        0 = Yes    / 1 = No           
                5               Snow/Ice Background Path             0 = Yes    / 1 = No           
                7, 6            Land or Water Path                   00 = Water                    
                                                                    01 = Coastal                  
                                                                    10 = Desert                   
                                                                    11 = Land                     
                                                                                                    
                Second Byte                                                                        
                --------------------------------------------------------------------------          
                                                                                                    
                1, 0            Sun-glint Under CTP Retrieval        00 = No CTP Ret.              
                                                                    01 = No Sun-glint             
                                                                    10 = Sun-glint                
                                                                                                    
                3, 2            Snow/Ice Under CTP Retrieval         00 = No CTP Ret.              
                                                                    01 = No Snow/Ice              
                                                                    10 = Snow/Ice                 
                                                                                                    
                6, 5, 4         Surface Type Under CTP Retrieval    000 = No CTP Ret.              
                                                                    001 = Water                    
                                                                    010 = Coast                    
                                                                    011 = Desert                   
                                                                    100 = Land                     
                                                                    101 = Other                    
                                                                                                    
                7               Day/Night Flag Under CTP Retrieval   01 = Day   
            ---------------------------
        Input is array([byte1, byte2]).
        Parameter options:
            cloud_mask_determined
            cloud_mask_value
            day_night: 1 day 0 night
        See pg 121 in MODIS User Guide.
    """
    output = np.empty(cloud_mask_flags.shape, np.float32)

    for (i,j),_ in np.ndenumerate(output):
        if parameter == 'day_night':
            if cloud_mask_flags[i,j][4] == '0': output[i,j] = 0
            else: output[i,j] = 1
        elif parameter == 'cloud_mask_determined':
            if cloud_mask_flags[i,j][7] == '0': output[i,j] = np.nan
            else: output[i,j] = 1
        elif parameter == 'sun_glint':
            if cloud_mask_flags[i,j][3] == '0': output[i,j] = 1
            else: output[i,j] = 0
        elif parameter == 'cloud_mask_value':
            cloud_mask_determined = cloud_mask_flags[i,j][7] 
            if cloud_mask_determined == '0': output[i,j] = np.nan 
            else:
                cloud_mask_value = cloud_mask_flags[i,j][5]+cloud_mask_flags[i,j][6]
                if cloud_mask_value == '00':
                    output[i,j] = 1
                else:
                    output[i,j] = 0
    return output
@njit(parallel=False)
def MODIS_tau_flag(tau_flags):
    """
        Input: 
            tau_flags: 2-D np.ndarray: first byte, formatted in binary strings

        Output: np.ndarray (2D) of shape tau_flags:
            1 where tau is nan,
            0 where tau retreival is good
        From data:
                ________________________
                                        Byte 3 -----------------------------------------------------------------           
                        0           Optical Thickness 1.6-2.1 General QA  0 = Not Useful                  
                                                                            1 = Useful                      
                        2,1         Optical Thickness 1.6-2.1 Condifence QA                               
                                                                            00 = No confidence              
                                                                            01 = Marginal                   
                                                                            10 = Good                       
                                                                            11 = Very Good                  
                        3           Effective Radius 1.6-2.1 General QA   0 = Not Useful                  
                                                                            1 = Useful                      
                        5,4         Effective Radius 1.6-2.1 Confidence QA                                
                                                                            00 = No confidence              
                                                                            01 = Marginal                   
                                                                            10 = Good                       
                                                                            11 = Very Good                  
                        7,6            Clear Sky Restoral Type QA                                         
                                                            00 = Not Restored                            
                                                            01 = Restored Via Edge detection             
                                                            10 = Restored Via Spatial  Variance          
                                                            11 = Restored Via 250m Tests                 
                        Byte 4 -----------------------------------------------------------------  
                ________________________
    """
    output = np.zeros(tau_flags.shape, np.int8)

    for (i,j),_ in np.ndenumerate(output):
        if tau_flags[i,j][7] == '0': # not useful
            output[i,j] = 1
        # tau_conf = tau_flags[i,j][5]+tau_flags[i,j][6]
        # if tau_conf == '00': # no confidence
        #     output[i,j] = 1
        # if tau_conf == '01': # marginal
        #     output[i,j] = 1
    return output
def EPIC_cloudmask_flag(cloud_mask, cloud_thresh):
    """
        Input: 
            cloud_mask: 2-D np.ndarray from EPIC_file['geophysical_data']['Cloud_Mask'][:]
            cloud_thresh: int, which threshold to consider cloudy
        Output:
            binary_mask: 2-D np.ndarray with 0 for not cloudy, 1 for cloudy, np.nan for
            non-Earth pixel
        
    For EPIC Cloud_Mask
        Cloud cloudiness flag (from data):
            description03:   0 -- non Earth pixel                                                              

            description04:   1 -- clear with high confidence                                                   

            description05:   2 -- clear with low confidence                                                    

            description06:   3 -- cloud with low confidence                                                    

            description07:   4 -- cloud with high confidence 
        Returns 1 if Confident cloudy, 0 if Confident or Probably Clear or probably cloudy, or nan if undetermined
    """
    binary_mask = np.copy(cloud_mask).astype(float) # cannot use np dtypes with clear_border
    binary_mask[cloud_mask>=cloud_thresh] = 1
    binary_mask[cloud_mask<cloud_thresh] = 0
    binary_mask[cloud_mask==0] = np.nan 
    return binary_mask
def VIIRS_cloudmask_flag(array, parameter):
    """
        Input: 
            cloud_mask: 2-D np.ndarray: [byte1,byte2]   (single pixel)
            parameter:  Which data field to look at. Options:
                'day_night'
                'cloud_mask_determined'
                'cloud_mask_value'
                'land_sea'

        Output: int or np.nan
                'day_night':    0 = Night  / 1 = Day / np.nan if error
                'cloud_mask_determined'     1 = Determined / np.nan = not determined
                'cloud_mask_value':     1 = Cloudy / 0 = Confident Clear, Probably Clear, or Uncertain 
                                        / np.nan if error / np.nan if undetermined
                'land_sea':       1 = Land 0 = Sea
    
    description04: Bit fields within each byte are numbered from the left:                             

    description05: 7, 6, 5, 4, 3, 2, 1, 0.                                                             

    description06: The left-most bit (bit 7) is the most significant bit.                              

    description07: The right-most bit (bit 0) is the least significant bit.                            

    description08:                                                                                     

    description09: Bit Field       Description                             Key                         

    description10: ---------       -----------                             ---                         

    description11:                                                                                     

    description12:  Byte 0 -----------------------------------------------------------                 

    description13:                                                                                     

    description14: 0               Cloud Mask Flag                      0 = Not  determined            

    description15:                                                      1 = Determined                 

    description16:                                                                                     

    description17: 2, 1            Unobstructed FOV Quality Flag        00 = Cloudy                    

    description18:                                                      01 = Uncertain                 

    description19:                                                      10 = Probably  Clear           

    description20:                                                      11 = Confident  Clear          

    description21:                  PROCESSING PATH                                                    

    description22:                 ---------------                                                     

    description23: 3               Day or Night Path                    0 = Night  / 1 = Day           

    description24: 4               Sunglint Path                        0 = Yes    / 1 = No            

    description25: 5               Snow/Ice Background Path             0 = Yes    / 1 = No            

    description26: 7, 6            Land or Water Path                   00 = Water                     

    description27:                                                      01 = Coastal                   

    description28:                                                      10 = Desert                    

    description29:                                                      11 = Land                      

    description30:  Byte 1 -----------------------------------------------------------                 

    description31:                                                                                     

    description32: 0        High Cloud Test 1.38um result                  0 = No / Not applied        

    description33:                                                         1 = Might have cloud        

    description34: 1        High Cloud Test 1.38um applied?                0 = Not applied             

    description35:                                                         1 = Applied                 

    description36: 2        Visible Reflectance Threshold test result      0 = No / Not applied        

    description37:                                                         1 = Might have cloud        

    description38: 3        Visible Reflectance Threshold test applied?    0 = Not applied             

    description39:                                                         1 = Applied                 

    description40: 4        R0.86 / R0.65um test result                    0 = No / Not applied        

    description41:                                                         1 = Might have cloud        

    description42: 5        R0.86 / R0.65um test applied?                  0 = Not applied             

    description43:                                                         1 = Applied   
    """
    byte1_string = format(array[0], '08b')   
    byte2_string = format(array[1], '08b') 
    if parameter == 'day_night':
        try:
            day_night = int(byte1_string[4], 2)
        except:
            day_night = np.nan
        return day_night
    elif parameter == 'cloud_mask_determined':
        try:
            cloud_mask_determined = int(byte1_string[7], 2)
        except:
            cloud_mask_determined = np.nan
        if cloud_mask_determined == 0:
            return np.nan
        else:
            return 1
    elif parameter == 'cloud_mask_value':
        try:
            cloud_mask_determined = int(byte1_string[7], 2)
        except: return np.nan 
        if cloud_mask_determined == 0: return np.nan 
        try:
            cloud_mask_value = int(byte1_string[5:7], 2)
        except: cloud_mask_value = np.nan
        if cloud_mask_value == 0:
            return 1
        else:
            return 0
def georing_edge_clouds_only(matrix):

    nanmask = np.isnan(matrix).astype(int)
    edge_mask =  (nanmask - uf.clear_one_border(uf.clear_one_border(nanmask, 'bottom'), 'top')).astype(bool)

    with_edge = matrix.copy()
    with_edge[edge_mask] = 1
    cleared = uf.clear_one_border(uf.clear_one_border(matrix, 'bottom'), 'top').astype(float)
    cleared[edge_mask] = np.nan
    cleared[np.isnan(matrix)] = np.nan

    # return cleared
    return (matrix - cleared).round() # fix floating point errors
def geolocate_POLDER(shape):
    """
    Create array of shape (1080,2160) of lon and lat, based on
    POLDER/Parasol Level-2 Product Data Format and User Manual
    NOTE: Creates wrong values for places outside of the projection.
    Need to be marked nan elsewhere.
    """
    if shape != (1080,2160): raise ValueError('Input shape is the wrong size. Is this the correct POLDER dataset?')

    def lon_lat_one_point(i,j):
        i, j = i+1,j+1
        lat = 90- (i-.5)/6
        N_i = int(round(1080*np.cos(lat*np.pi/180)))
        lon = (180/N_i)* (j-1080.5)

        return lon, lat

    lon, lat = np.zeros(shape), np.zeros(shape)

    for (i,j),_ in np.ndenumerate(lon):
        lon[i,j], lat[i,j] = lon_lat_one_point(i,j)

    return lon, lat
@njit()
def flip_POLDER_data(array):
    """
        Input: 2D np.ndarray (1080,2160) of any data from POLDER
        Output: 2D np.ndarray (1080,2160), centered on the opposite meridian

        POLDER data is projected on a sinusoidal equal-area projection, originally
        centered on Greenwich meridian. This function flips the projection so it is
        centered on the dateline instead.

        See Appendix B of the POLDER user manual
    """
    if array.shape != (1080,2160): raise ValueError('Input array is the wrong size. Is this the correct POLDER dataset?')
    flipped = np.empty(array.shape)
    for (i,j), _ in np.ndenumerate(array):
        i, j = i+1, j+1
        N_i = int(round(1080*np.sin(((i-0.5)/6)*np.pi/180)))
        new_j = 1081 - N_i + (j+2*N_i-1081) % (2*N_i)

        i,j,new_j = i-1,j-1, new_j-1
        flipped[i, new_j] = array[i,j]

    return flipped


if __name__ == '__main__':
    main()

