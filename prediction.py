#%%
! pip install ai-models
! pip install ai-models-fourcastnet
! pip install ai-models-panguweather

#%%
prediction_date = '20240501'
fourcastnet_output_path = f'{prediction_date}-fourcastnet.grib'
panguweather_output_path = f'{prediction_date}-panguweather.grib'
lead_time = 30 # 30 / 6 = 5 steps

#%%
! ai-models --download-assets --lead-time {lead_time} fourcastnet --input cds --date {prediction_date} --path {fourcastnet_output_path}
! ai-models --download-assets --lead-time {lead_time} panguweather --input cds --date {prediction_date} --path {panguweather_output_path}