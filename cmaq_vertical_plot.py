# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:36:21 2022

@author: schao
"""
import os
import re
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from matplotlib import colors
from concurrent.futures import ThreadPoolExecutor
from matplotlib import rcParams
config = {
    "font.family":'STSong',
    "font.size": 10,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

# 显示中文
plt.rcParams["font.family"]="STSong"

def chkdir(path):
	'''检查路径'''
	if not os.path.exists(path):
		os.mkdir(path)

def julian_to_date(tflag):
	'''
	purpose:根据tflag计算具体日期，tflag：2022256
	'''
	year = int(str(tflag)[0:4])
	day = int(str(tflag)[4:])
	firstDay = datetime(year,1,1)
	theDay = firstDay+timedelta(day-1)
# 	calDay = datetime.strftime(theDay,'%Y%m%d')
	return theDay

def decide_index(lon_array,lon_p):
	'''
	purpose:根据给定的经度数组和经度坐标，确定该经度坐标在数组中的位置
	'''
	lonL = lon_array.tolist()
	lonL.append(lon_p)
	idx = sorted(lonL).index(lon_p)
	return idx

def dcmap(cmp_file):
	'''
	purpose:读取ncl的rgb文件，转换为python可用格式
	'''
	fid = open(cmp_file)
	data = fid.readlines()
	n = len(data[2:])
	rgb = np.zeros((n,3))
	for i in np.arange(n):
		rgb[i][0]=data[i+2].split(' ')[0]
		rgb[i][1]=data[i+2].split(' ')[1]
		rgb[i][2]=data[i+2].split(' ')[2]
		icmap=colors.ListedColormap(rgb,name='my_color')
	return icmap

def get_wrf_dat(wrf_file):
	'''
	purpose:获取wrf数据，需要计算位势高度的变量
	args:
		wrf_file:wrf文件
	'''
	with xr.open_dataset(wrf_file) as dw:
# 		keys = list(dw.keys())
		XLAT = dw['XLAT'][0,:,:]
		XLON = dw['XLONG'][0,:,:]
		HGT = dw['HGT'][0,:,:]
		PH = dw['PH'][0,:,:,:]
		PHB = dw['PHB'][0,:,:,:]
	return XLAT,XLON,HGT,PH,PHB

def ctm_lev_to_wrf_index(cctm_file):
	'''
	purpose:获取cctm污染数据的高度层对应wrf高度层的索引
	'''
	wrf_eta = [1.0000, 0.9975, 0.9950, 0.9900, 0.9800,
		0.9700, 0.9600, 0.9400, 0.9200, 0.9000,
        0.8750, 0.8500, 0.8200, 0.7900, 0.7550,
        0.7200, 0.6850, 0.6500, 0.6150, 0.5800,
        0.5450, 0.5100, 0.4750, 0.4400, 0.4000,
        0.3600, 0.3200, 0.2800, 0.2400, 0.2000,
        0.1600, 0.1200, 0.0800, 0.0400, 0.0000]
	wrf_eta = np.array(wrf_eta,dtype = 'float32')
	wrf_eta = list(wrf_eta)
	with xr.open_dataset(cctm_file) as dc:
		ctm_lev = list(dc.attrs['VGLVLS'])
	cctm_index = []
	for i in ctm_lev:
		ind = wrf_eta.index(i)
		cctm_index.append(ind)
	return cctm_index

def read_site(site_file):
	'''
	purpose:读取站点信息
	'''
	colnames = ['site_mes','east_index','north_index','col','row']
	df = pd.read_table(site_file,sep=',',engine='python',skiprows=2,names=colnames)
	return df

def get_height_from_wrf(cctm_file,wrf_file,north_index,east_index):
	'''
	purpose:根据eta层，从wrf中提取对应站点的海拔高度和对应站点的经度数组
	'''
	cctm_index = ctm_lev_to_wrf_index(cctm_file)
	XLAT,XLON,HGT,PH,PHB = get_wrf_dat(wrf_file)
	lon = np.array(XLON[north_index,:])
# 	print(lon)
	gmp = []
	for i in cctm_index:
		hgt = HGT[north_index,east_index]
		ph = PH[i,north_index,east_index]
		phb = PHB[i,north_index,east_index]
		ht = np.array((ph+phb)/9.81-hgt)
# 		print(ht)
		gmp.append(ht)
	return gmp,lon

def plot_pol_from_cmaq(cctm_file,cmp_file,north_index,gmp,lon,lon1,lon2,site_name,species_name,outdir):
	'''
	purpose:从cmaq中提取对应污染物，对应站点的探空高度的污染数据
	'''
	lonP1 = decide_index(lon, lon1)
	lonP2 = decide_index(lon, lon2)
	gmp = [str('%d'%g) for g in gmp]
	gmp = gmp[:-1]
# 	print(len(gmp),len(lon))
	with xr.open_dataset(cctm_file) as ctd:
# 		keys = list(ctd.keys())
		tFlag = np.array(ctd['TFLAG'][:,0,0])
# 		print(keys)
		if species_name == 'PM2_5':
			f1 = np.array(ctd['ASO4I'][:,:,:,:])
			f2 = np.array(ctd['ASO4J'][:,:,:,:])
			f3 = np.array(ctd['ANH4I'][:,:,:,:])
			f4 = np.array(ctd['ANH4J'][:,:,:,:])
			f5 = np.array(ctd['ANO3I'][:,:,:,:])
			f6 = np.array(ctd['ANO3J'][:,:,:,:])
			f7 = np.array(ctd['AECI'][:,:,:,:])
			f8 = np.array(ctd['AECJ'][:,:,:,:])
			f9 = np.array(ctd['ACAJ'][:,:,:,:])
			f10 = np.array(ctd['AMGJ'][:,:,:,:])
			f11 = np.array(ctd['AKJ'][:,:,:,:])
			f12 = np.array(ctd['AALJ'][:,:,:,:])
			f13 = np.array(ctd['AFEJ'][:,:,:,:])
			f14 = np.array(ctd['ASIJ'][:,:,:,:])
			f15 = np.array(ctd['ATIJ'][:,:,:,:])
			f16 = np.array(ctd['AMNJ'][:,:,:,:])
			f17 = np.array(ctd['AOTHRI'][:,:,:,:])
			f18 = np.array(ctd['AOTHRJ'][:,:,:,:])
			f19 = np.array(ctd['ANAI'][:,:,:,:])
			f20 = np.array(ctd['ANAJ'][:,:,:,:])
			f21 = np.array(ctd['ACLI'][:,:,:,:])
			f22 = np.array(ctd['ACLJ'][:,:,:,:])
			f23 = np.array(ctd['AXYL1J'][:,:,:,:])
			f24 = np.array(ctd['AXYL2J'][:,:,:,:])
			f25 = np.array(ctd['AXYL3J'][:,:,:,:])
			f26 = np.array(ctd['ATOL1J'][:,:,:,:])
			f27 = np.array(ctd['ATOL2J'][:,:,:,:])
			f28 = np.array(ctd['ATOL3J'][:,:,:,:])
			f29 = np.array(ctd['ABNZ1J'][:,:,:,:])
			f30 = np.array(ctd['ABNZ2J'][:,:,:,:])
			f31 = np.array(ctd['ABNZ3J'][:,:,:,:])
			f32 = np.array(ctd['AISO1J'][:,:,:,:])
			f33 = np.array(ctd['AISO2J'][:,:,:,:])
			f34 = np.array(ctd['AISO3J'][:,:,:,:])
			f35 = np.array(ctd['ATRP1J'][:,:,:,:])
			f36 = np.array(ctd['ATRP2J'][:,:,:,:])
			f37 = np.array(ctd['ASQTJ'][:,:,:,:])
			f38 = np.array(ctd['AALKJ'][:,:,:,:])
			f39 = np.array(ctd['AORGCJ'][:,:,:,:])
			f40 = np.array(ctd['AOLGAJ'][:,:,:,:])
			f41 = np.array(ctd['AOLGBJ'][:,:,:,:])
			f42 = np.array(ctd['APOCI'][:,:,:,:])
			f43 = np.array(ctd['APOCJ'][:,:,:,:])
			f44 = np.array(ctd['APNCOMI'][:,:,:,:])
			f45 = np.array(ctd['APNCOMJ'][:,:,:,:])
			species = f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+\
				f18+f19+f20+f21+f22+f23+f24+f25+f26+f27+f28+f29+f30+f31+f32+f33+\
					f34+f35+f36+f37+f38+f39+f40+f41+f42+f43+f44+f45
		elif species_name == 'O3':
			species = np.array(ctd[species_name][:,:,:,:])*1963
		else:
			species = np.array(ctd[species_name][:,:,:,:])
	thisday = julian_to_date(tFlag[0])
	delta = (lon[lonP2]-lon[lonP1])/5
	lon_x = list(np.arange(lon[lonP1],lon[lonP2],delta))
# 	lon_x.append(lon[lonP2])
	lon_ticks = [str('%.3f'%i)+'E' for i in lon_x]
	icmap = dcmap(cmp_file)
	chkdir(outdir)
	speciesName = {'PM2_5':'$\mathrm{PM}_\mathrm{2.5}$','O3':'$\mathrm{O}_\mathrm{3}$'}
	species_name2 = speciesName.get(species_name)
	for i in range(species.shape[0]):
		data = species[i,:,north_index,lonP1:lonP2]
		thisTime = thisday+timedelta(hours=i+8)
		tTime = thisTime.strftime('%Y%m%d_%H:00:00')
		print('>>> ',species_name,' ',tTime,' ',site_name)

		plt.contourf(lon[lonP1:lonP2],gmp,data,cmap = icmap)
		plt.colorbar(label='CMAQ模拟浓度('+r'$\mu$g/m$^{3}$)')
		plt.xticks(lon_x,lon_ticks)
		plt.ylabel('高度(m)')
		plt.title(species_name2+' 垂直层浓度分布'+'\n')
		plt.title('纬度：'+site_name,loc='left')
		plt.title('时间：'+str(tTime),loc='right')
		plt.tight_layout()
		plt.savefig(outdir+'/'+species_name+'_'+site_name+'_'+str(tTime[:11])+'.png',dpi=500)
		plt.close()

def main_plot(cctmPath,wrfPath,sitePath,cmpFile,outdir,thistime,lon1,lon2,time_range):
	'''
	purpose:串联运行主程序
	'''
	speciesArray = ['PM2_5','O3']
	df = read_site(sitePath+'/'+'xiamen2.txt')
	cctm_file = cctmPath+'/'+'CCTM_saprc07.ACONC.xiamen12_'+thistime
	thistime2 = thistime[0:4]+'-'+thistime[4:6]+'-'+thistime[6:8]
	wrf_file = wrfPath+'/'+'xiamen12.'+time_range+'.wrfout_d02_'+thistime2+'_12_00_00'
# 	wrf_file = wrfPath+'/'+'wrfout_d02_'+thistime2+'_12:00:00'
	print(wrf_file)
	for species_name in speciesArray:
		for i in range(len(df['north_index'][:])):
			north_index = df['north_index'][i]
			east_index = df['east_index'][i]
			site_name = re.split(r"[ ]+",df['site_mes'][i])[2]
			gmp,lon = get_height_from_wrf(cctm_file, wrf_file, north_index,east_index)
			plot_pol_from_cmaq(cctm_file, cmpFile, north_index, gmp, lon, lon1, lon2, site_name, species_name, outdir)

if __name__ == '__main__':
	cctmPath = 'D:/jobs/聚光科技/谱育科技/202209/vertical_data'
	wrfPath = 'D:/jobs/聚光科技/谱育科技/202209/vertical_data'
	sitePath = 'D:/jobs/聚光科技/谱育科技/202209/vertical_data'
	cmpFile = 'D:/jobs/聚光科技/谱育科技/202209/vertical_data/MPL_Blues.rgb'
	starttime = '20221005'
	endtime = '20221005'
	outdir = 'D:/jobs/聚光科技/谱育科技/202209/vertical_data/'+starttime
	yesterday = datetime.strptime(starttime, '%Y%m%d')+timedelta(days=-1)
	preDay = datetime.strptime(starttime, '%Y%m%d')+timedelta(days=11)
	time_range = yesterday.strftime('%Y%m%d')+'12_'+preDay.strftime('%Y%m%d')+'06'
	lon1,lon2 = 117.3,118.7
	STime = datetime.strptime(starttime, '%Y%m%d')
	ETime = datetime.strptime(endtime, '%Y%m%d')
	timeList= []
	while STime <= ETime:
		thistime = STime.strftime('%Y%m%d')
		timeList.append(thistime)
		main_plot(cctmPath, wrfPath, sitePath, cmpFile, outdir, thistime, lon1, lon2,time_range)
		STime += timedelta(days=1)
# 	with ThreadPoolExecutor(max_workers=12) as executer:
# 		all_work = {executer.submit(main_plot, cctmPath, wrfPath, sitePath, cmpFile, outdir, thistime, lon1, lon2,time_range):
# 			thistime for thistime in timeList}
