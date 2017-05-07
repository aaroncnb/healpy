import numpy as np
import healpy as hp
from haperflux import haperflux

### 'radius', 'rinner', and 'router' should be given in arcminutes

def healpix_phot(inputlist, maplist, radius, rinner, router, galactic=True, decimal=True):
    ##Here is the "observation data structure", which just means "a bunch of details 
    ## about the different all-sky data sources which could be used here.
    freqlist =     ['30','44','70','100','143','217','353','545','857','1874','2141','2998','3331','4612','4997','11992','16655','24983','33310']
    freqval =      [28.405889, 44.072241,70.421396,100.,143.,217.,353.,545.,857.,1874.,2141.,2998.,3331.,4612.,4997.,11992.,16655.,24983.,33310.]
    fwhmlist =     [33.1587,28.0852,13.0812,9.88,7.18,4.87,4.65,4.72,4.39,0.86,0.86,4.3,0.86,0.86,4,3.8,0.86,3.8,0.86] # fwhm in arcminutes
    band_names =   ["akari9", "iras12", "akari18","iras25","iras60","akari65","akari90","iras100","akari140","akari160","planck857", "planck545"]
    band_centers = [ 60e-6,    65e-6,    90e-6,   100e-6,   140e-6,    160e-6,    350e-6,      550e-6]

    

    k0 = 1.0 
    k1 = rinner/radius 
    k2 = router/radius 
    apcor = ((1 - (0.5)**(4*k0**2))-((0.5)**(4*k1**2) - (0.5)**(4*k2**2)))**(-1)
  
    # 'galactic' overrules 'decimal' 
    if (galactic==True):
        dt=[('sname',np.dtype('S13')),('glon',np.float32),('glat',np.float32)]
        targets = np.genfromtxt(inputlist, delimiter=",",dtype=dt)

    ns = len(targets['glat'])

    fd3 = -1
    fd_err3 = -1

    fn = np.genfromtxt(maplist, delimiter=" ", dtype='str') 
    nmaps = len(fn)
    ## Initialize the arrays which will hold the results
    fd_all = np.zeros((ns,nmaps))
    fd_err_all = np.zeros((ns,nmaps))
    fd_bg_all = np.zeros((ns,nmaps))
    #openw,1,file_basename(inputlist+'.photo'),width=200

    #if radius == None:
    #    printf 1,'; A multiplicative aperture correction factor of ',apcor,' has been applied to the flux densities'
    #    printf 1, '; assuming that the source is a point source. Flux densities are in Jy.'
    #elif:
    #    printf 1,';No aperture correction factor has been applied. Flux densities are in Jy.'

    #printf 1, ';'
    #printf 1, ';Output format:'
    #printf 1, '; Source_Name  Map_number  GLON   GLAT   Flux (Jy) Flux_RMS (Jy) Median_Background_Flux (Jy)'
    #printf  1, ';'
    #printf  1, '; Map List:'
    # Make a header-legend that says what order the maps were processed:
    #for i = 0, nmaps-1:
    #    printf, 1, '; Map #',i, fn[i]
    #
    #printf, 1, ';'
    
    # Start the actual processing: Read-in the maps.
    for ct2 in range(0,nmaps):
        xtmp_data, xtmp_head = hp.read_map(fn[ct2], memmap=True, h=True, verbose=False, nest=False)
        freq = dict(xtmp_head)['FREQ']
        units = dict(xtmp_head)['TUNIT1']
        freq_str = str(freq)
        idx = freqlist.index(str(freq))
        currfreq = int(freq)
        
        if (radius == None):
            radval = fwhmlist[idx]
        else:
            radval = radius
        
        
        for ct in range(0,ns): 
            

            glon = targets['glon'][ct]
            glat = targets['glat'][ct]

            fd_all[ct,ct2], fd_err_all[ct,ct2], fd_bg_all[ct,ct2] = \
                haperflux(inmap= xtmp_data, freq= currfreq, lon=glon, lat=glat, \
                        res_arcmin= radius, aper_inner_radius=radius, aper_outer_radius1=rinner, \
                        aper_outer_radius2=router,units=units)
            
            #print "T: "+str(ct+1)+", nu: "+freq_str+", Flux Density (Jy) "+str(round(fd_all[ct,ct2]))+", Error (Jy) "+str(round(fd_err_all[ct,ct2]))+" Background (Jy) "+str(round(fd_bg_all[ct,ct2]))
                 

            #print np.isfinite(fd_err_all[ct,ct2])
            if (np.isfinite(fd_err_all[ct,ct2]) == False):
                fd_all[ct,ct2] = -1
                fd_err_all[ct,ct2] = -1
            else:
                if radius==None:
                    fd_all[ct,ct2] = fd_all[ct,ct2]*apcor
                    fd_err_all[ct,ct2] = fd_err_all[ct,ct2]*apcor
    print fd_all  
    return fd_all, fd_err_all, fd_bg_all

