#
#  This file is part of Healpy.
#
#  Healpy is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  Healpy is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Healpy; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
#  For more information about Healpy, see http://code.google.com/p/healpy
#
"""Does aperture photometry on a Healpix map with a circle of a given
   radius, subtracting the background determined from an annulus.
Units are assumed to be mK_RJ and are converted to Jy.

 HISTORY

26-Jun-2010  C. Dickinson   1st go
 25-Jul-2010  C. Dickinson   Added conversion option from T_CMB->T_RJ
 25-Jul-2010  C. Dickinson   Added 2nd outer radius
 26-Aug-2010  C. Dickinson   Added generic 'unit' option
 02-Sep-2010  C. Dickinson   Tidied up the code a little
 19-Oct-2010  C. Dickinson   Use ang2vec rather than long way around
                             via the pixel number
 20-Oct-2010  M. Peel        Fix MJy/Sr conversion; add aliases for
							  formats
							  excluding
							  underscores

 10-Feb-2011  C. Dickinson   Added column option to allow polarization
 16-Feb-2011  C. Dickinson   Added /dopol keyword for doing polarized intensity
 12-Mar-2011  C. Dickinson   Added flexibility of reading in file or array
 01-Jun-2011  C. Dickinson   Added noise_model option
 10-Nov-2011  M. Peel        Adding AVG unit option.
 20-Sep-2012  P. McGehee     Ingested into IPAC SVN, formatting changes
 11-Apr-2016  A. Bell        Translated to Python
------------------------------------------------------------
"""

import numpy as np
from . import *
import healpy.projector as pro
import astropy.io.fits as fits
import sys
import math
from astropy.stats import mad_std
from astropy import units as u
from astropy.coordinates import SkyCoord



def removeCommonElements(outerpix1, outerpix2):
    for pix in outerpix2[:]:
        if pix in outerpix1:
            outerpix2.remove(pix)
            outerpix1.remove(pix)

def removeCommonElementsNumpy(outerpix1, outerpix2):
    outpix1 = outerpix1.copy()
    outpix2 = outerpix2.copy()
    for pix in outerpix2[:]:
        if pix in outerpix1:
            np.delete(outpix1,pix)
            np.delete(outpix2,pix)
    return outpix1, outpix2

def deleteCommon(outerpix1,outerpix2):
    outerpix = np.delete(outerpix2, outerpix1)
    return outerpix


def planckcorr(freq):
    h = 6.62606957E-34
    k = 1.3806488E-23
    T = 2.725
    x = h*(freq*1e9)/k/T
    return (exp(x) - 1)**2/x**2/exp(x)


def convertToJy(units, thisfreq, npix=None, pix_area = None):

    # get conversion factors for an aperture to Jy/pix, from any of the following units:
    # Kelvin(RJ), Kelvin(CMB), MJy/sr, Average
    #print "Getting the appropriate conversion factors..."

    if pix_area != None and npix != None:
        print "Only one of npix or pix_area should be specified! Proceeding with npix."


    if pix_area == None:
        pix_area = 4.*np.pi / npix

    factor = 1.0

        if (units == 'K') or (units == 'K_RJ') or (units == 'KRJ'):
            factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area

        elif (units == 'mK') or (units == 'mK_RJ') or (units == 'mKRJ'):
            factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area / 1.0e3

        elif (units == 'uK') or (units == 'uK_RJ') or (units == 'uKRJ'):
            factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area / 1.0e6

        elif (units == 'K_CMB') or (units == 'KCMB'):
            factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area / planckcorr(thisfreq)

        elif (units == 'mK_CMB') or (units == 'mKCMB'):
            factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area / 1.0e3 / planckcorr(thisfreq)

        elif (units == 'uK_CMB') or (units == 'uKCMB'):
            factor = 2.*1381.*(thisfreq*1.0e9)**2/(2.997e8)**2 * pix_area / 1.0e6 / planckcorr(thisfreq)

        elif (units == 'MJy/sr') or (units == 'MJY/SR') or (units == 'MjySr'):
            factor = pix_area * 1.0e6

        elif (units == 'Jy/pixel') or (units == 'JY/PIXEL') or (units == 'JY/PIX') or (units == 'JyPix'):
            factor = 1.0

        elif (units == 'average') or (units == 'avg') or (units == 'Average') or (units == 'AVG'):
            factor = 1.0 / float(ninnerpix)
        else:
            print "Invalid units specified! "

        return factor

### In the next pass, let's use the astropy system for coordinate conversion
### The IDL EULER routine is the last remaining bit of Python wrapped IDL code here
### This is a good guide for the astropy implementation:   http://docs.astropy.org/en/stable/coordinates/



def haperflux(inmap, freq, lon, lat, res_arcmin, aper_inner_radius, aper_outer_radius1, aper_outer_radius2, \
              units, fd=0, fd_err=0, fd_bg=0, column=0, dopol=False, nested=False, noise_model=0, centroid=False, arcmin=True):

    '''Perfurms the circular aperture source flux measurement, as well as background
    subtraction measured by pixels in a specified annulus.

    Parameters
    ----------
    inmap : float, array-like or str
     Healpix fits file or map array. If an array, it assumed to be 'RING'
     ordering if ordering keyword not set.
    freq : float
     Frequency (GHz) to allow conversion to Jy.
    res_arcmin : float
     Angular resolution (arcmin FWHM) of the data This is needed for
     the error estimation and only needs to be approximate( for noise_model=1 only )
    lon : float
     Longitude of aperture (deg)
    lat : float
     Latitude of aperture (deg)
    aper_inner_radius : float
     Inner radius of aperture (arcmin)
    aper_outer_radius1 : float
     1st Outer radius of aperture beteween aperture and  b/g
     annulus (arcmin). Make this the same as the inner radius for single annulus
     with no gap.
    aper_outer_radius2 : float
     2nd Outer radius of aperture for b/g annulus (arcmin)
    units : str
     String defining units in the map. Options: ['K','K_RJ', 'K_CMB', 'MJy/sr','Jy/pixel'].
     Default is 'K_RJ'. m for milli and u for micro can also be used for K e.g. 'mK_RJ' or 'mK'
    column : int,  optional
     This can be set to any integer which represents the column of the map
     (default is column=0.)
    #dopol : bool, optional [Feature from original IDL version. Not yet implemented.]
    #  If this keyword is set, then it will calculate the
    #  polarized intensity from columns 1 and 2 (Q and
    #  U) as PI=sqrt(Q^2+U^2) with *no noise bias* correction
    #  N.B. This overrides the column option
    nested : bool
     If True, then the ordering is NESTED. Default is to assume RING if not reading in a file)
    noise_model : int
     Noise model for estimating the uncertainty.
     0 (DEFAULT) = approx uncertainty for typical/bg annulus aperture sizes (only approximate!).
     1 = assumes white uncorrelated noise (exact) and will under-estimate in most cases with real backgrounds!
    # centroid : [Not yet impleme]
    #  reproject the source using gnomdrizz and uses the centroid of the source instead of input coordinates
    #  0 (DEFAULT) = no centroiding is done
    #  1 = centroiding performed using IDL code cntrd

    Returns
    -------
     fd : array-like
      Integrated flux density in aperture after background subtraction (Jy)
     fd_err : array-like
      Estimate of error on integrated flux density (Jy)
     fd_bg : array-like
       Background flux density estimate (Jy)
    '''

#check parameters
    if len(sys.argv) > 8:
        print ''
        print 'SYNTAX:-'
        print ''
        print 'haperflux(inmap, freq, res_arcmin, lon, lat, aper_inner_radius, aper_outer_radius1, aper_outer_radius2, units, fd, fd_err, fd_bg,column=column, noise_model=noise_model, /dopol, /nested)'
        print ''
        exit()


    #set parameters
    inmap = inmap
    thisfreq = float(freq) # [GHz]
    lon = float(lon) # [deg]
    lat = float(lat) # [deg]
    aper_inner_radius  = float(aper_inner_radius)  # [arcmin]
    aper_outer_radius1 = float(aper_outer_radius1) # [arcmin]
    aper_outer_radius2 = float(aper_outer_radius2) # [arcmin]

    # Convert the apertures to radians, if given in arcminutes.
    #     The pixel-finder 'query_disc' expects radians:
    if arcmin== True:

        aper_inner_radius  = aper_inner_radius/60.*np.pi/180.  # [rad]
        aper_outer_radius1 = aper_outer_radius1/60.*np.pi/180. # [rad]
        aper_outer_radius2 = aper_outer_radius2/60.*np.pi/180. # [rad]

    # read in data
    s = np.size(inmap)

    if (s == 1):
        print "Filename given as input..."
        print "Reading HEALPix fits file into a numpy array"

        hmap,hhead = hp.read_map(inmap, hdu=1,h=True, nest=nested, memmap=True) #Check if Ring or Nested is needed
        #http://healpy.readthedocs.org/en/latest/generated/healpy.fitsfunc.read_map.html
        #print np.size(hmap)

    if (s>1):
        #print "Numpy HEALPix array given as input, proceeding..."
        hmap = inmap


    if (nested==False):
        ordering='RING'
    else:
        ordering ='NESTED'

    nside = np.sqrt(len(hmap)/12)
    if (round(nside,1)!=nside) or ((nside%2)!=0):
        print ''
        print 'Not a standard Healpix map...'
        print ''
        exit()


    npix = 12*nside**2
    ncolumn = len(hmap)

# set column number and test to make sure there is enough columns in
# the file!
    if (column == 0):
        column = 0

    else:
        column=round(column,1)

    if (((column+1) > ncolumn) and (dopol== 0)):
        print ''
        print 'Column number requested larger than the number of columns in the file!'
        print ''
        exit()


    # check for dopol keyword for calculating polarized intensity
    if (dopol==0):
        dopol = 0
    else:
        dopol = 1
        column = 1
        if (ncolumn < 3):
            print ''
            print 'To calculate polarized intensity (PI), requires polarization data with 3 columns or more...'
            print ''
            exit()

    #-----do the centroiding here

    if (centroid==True):
        print 'Doing Re-centroiding of coordinates'


    # get pixels in aperture
    phi   = lon*np.pi/180.
    theta = np.pi/2.-lat*np.pi/180.
    vec0  = hp.ang2vec(theta, phi)

    # According to the HP git repository- hp.query_disc is faster in RING

    #print "Getting the innermost (source) pixel numbers"
    ## Get the pixels within the innermost (source) aperture
    innerpix = hp.query_disc(nside=nside, vec=vec0, radius=aper_inner_radius, nest=nested)

    #print "Getting the background ring pixel numbers"
    ## Get the pixels within the inner-ring of the background annulus
    outerpix1 = hp.query_disc(nside=nside, vec=vec0, radius=aper_outer_radius1, nest=nested)
    #nouterpix1 = len(outerpix1)

    ## Get the pixels within the outer-ring of the background annulus
    outerpix2 = hp.query_disc(nside=nside, vec=vec0, radius=aper_outer_radius2, nest=nested)
    #nouterpix2 = len(outerpix2)


# Identify and remove the bad pixels
# In this scheme, all of the bad pixels should have been labeled with HP.UNSEEN in the HEALPix maps

    #print "Checking for bad pixels"
    bad0 = np.where(hmap[innerpix] == hp.UNSEEN)
    #print "Printing bad0"
    #print bad0
    innerpix_masked = np.delete(innerpix,bad0)
    ninnerpix = len(innerpix_masked)

    bad1 = np.where(hmap[outerpix1] == hp.UNSEEN)
    outerpix1_masked = np.delete(outerpix1,bad1)
    nouterpix1 = len(outerpix1_masked)

    bad2 = np.where(hmap[outerpix2] == hp.UNSEEN)
    outerpix2_masked = np.delete(outerpix2,bad2)
    nouterpix2 = len(outerpix2_masked)

    #print str(bad0)+"bad pixels found."

    if (ninnerpix == 0) or (nouterpix1 == 0) or (nouterpix2 == 0):
        print ''
        print '***No good pixels inside aperture!***'
        print ''
        fd = np.nan
        fd_err = np.nan
        exit()

    innerpix  = innerpix_masked
    outerpix1 = outerpix1_masked
    outerpix2 = outerpix2_masked

    # find pixels in the annulus (between outerradius1 and outeradius2)
    # In other words, remove pixels of Outer Radius 2 that are enclosed within Outer Radius 1

    bgpix = np.delete(outerpix2, outerpix1)
    #print "Common Elements Removed"
    #print str(len(bgpix))+" background pixels used."
    #print "Printing background pixel list:"
    #print bgpix

    nbgpix = len(bgpix)



    factor = covnertToJy(units, thisfreq, npix)
# override columns if /dopol keyword is set

    if (dopol == 1):
        #print "Doing the flux calculations (using polarization data)"
        ncalc = 2

    else:
        ncalc = 1
        #print "Doing the flux calculations (using only T, or intensity/flux data)"

    for i in range(0, ncalc):

        #print "ncalc = "+str(ncalc)

        if (dopol == 1):
            column=i

        # Get pixel values in inner radius, converting to Jy/pix

        fd_jypix_inner = hmap[innerpix] * factor

        # sum up integrated flux in inner
        #print "Total flux of the source aperture:"
        fd_jy_inner = np.sum(fd_jypix_inner)
        #print str(fd_jy_inner)+" Jy "

        # same for outer radius but take a robust estimate and scale by area

        fd_jy_outer = np.median(hmap[bgpix]) * factor
        #print "Robust (median) estimate of the background:"+str(fd_jy_outer)+" Jy/pix "

        # subtract background

        fd_bg        = fd_jy_outer
        fd_bg_scaled = fd_bg*float(ninnerpix)
        fd           = fd_jy_inner - fd_bg_scaled

        #print "Scaled Background Level: "+str(fd_bg_scaled)+" Jy"
        #print "Source Background ratio: "+str(fd_jy_inner/fd_bg_scaled)


        #estimate error based on robust sigma of the background annulus
        # new version (2-Dec-2010) that has been tested with simulations for
        # Planck early paper and seems to be reasonable for many applications

        if (noise_model == 0):
            #print "Usiing the background annuus stddev as the noise estimate"
            Npoints = (pix_area*ninnerpix) /  (1.13*(float(res_arcmin)/60. *np.pi/180.)**2)
            Npoints_outer = (pix_area*nbgpix) /  (1.13*(float(res_arcmin)/60. *np.pi/180.)**2)
            fd_err = np.std(hmap[bgpix]) * factor * ninnerpix / math.sqrt(Npoints)
            #print "Noise level of the source: "+str(fd_err)+" Jy"


        # works exactly for white uncorrelated noise only!
        if (noise_model == 1):
            #print "Using the robust sigma (median absolute deviation) noise model"
            k = np.pi/2.

            fd_err = factor * math.sqrt(float(ninnerpix) + (k * float(ninnerpix)**2/nbgpix)) * mad_std(hmap[bgpix])
                #astropy.stats's mad_std seems to have the same functionality as IDLastro's "Robust_Sigma"

        # if dopol is set, then store the Q estimate the first time only
        if(dopol == 1) and (i == 1):
            fd1 = fd
            fd_err1 = fd_err
            fd_bg1 = fd_bg



        # if dopol is set, combine Q and U to give PI
    if (dopol == 1):
        fd = math.sqrt(fd1**2 +fd**2)
        fd_err = math.sqrt( 1./(fd1**2 + fd**2) * (fd1**2*fd_err1**2 + fd**2*fd_err** 2))
        fd_bg = math.sqrt(fd_bg1**2 + fd_bg**2)


    return fd, fd_err, fd_bg_scaled


### 'radius', 'rinner', and 'router' should be given in arcminutes

def aperphot(inputlist, maplist, radius, rinner, router, galactic=True, decimal=True):
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
