import matplotlib.pyplot as plt
import numpy as np
import scipy


def extend_angles(angles, ext):
    """
    Extends angles from the range (0, 360) to (0 - ext, 360 + ext).

    Parameters
    ----------
    angles : array-like or Angles object
        Array of angles in degrees.
    ext : int
        Extension in degrees beyond the (0, 360) range.

    Returns
    -------
    numpy.ndarray
        Extended array of angles.
    """
    # Extract the array if it's wrapped in an Angles object
    if hasattr(angles, 'arr'):
        angles = angles.arr

    num = ext // 360
    rem = ext % 360

    pre_angles = np.concatenate([angles - (i + 1) * 360 for i in range(num)])
    post_angles = np.concatenate([angles + (i + 1) * 360 for i in range(num)])

    first_angles = angles[angles >= 360 - rem] - (num + 1) * 360
    last_angles = angles[angles < rem] + (num + 1) * 360
    
    ext_ang = np.concatenate((first_angles, pre_angles, angles, post_angles, last_angles))
    return ext_ang


def extend_angles_2D(x,y,ext=1):
    """
    Extends a 2D distribution of angles beyond the range (0, 360), in 'ext'
    multiples of 360.

    Parameters
    ----------
    x, y : array-like
        Arrays of angles in degrees.
    ext : int, optional
        Multiple of 360 to extend the distribution by. Default is 1.

    Returns
    -------
    tuple of numpy.ndarray
        Extended x and y arrays.
    """
    x0 = np.copy(x)
    y0 = np.copy(y)

    x1 = np.copy(list(x0)*(2*ext + 1))
    y1 = np.copy(extend_angles(y0,360*ext))
    
    x = np.copy(extend_angles(x1,360*ext))
    y = np.copy(list(y1)*(2*ext + 1))
    return(x,y)


def rot(x,y,ang):
    """
    Rotates a 2D distribution about the point (180, 180).

    Parameters
    ----------
    x, y : array-like
        Arrays of points to be rotated.
    ang : float
        Angle in degrees to rotate the distribution.

    Returns
    -------
    tuple of numpy.ndarray
        Rotated x and y arrays.
    """
    x_ = np.copy(x)
    y_ = np.copy(y)
    ang=ang*np.pi/180
    x_-=180
    y_-=180
    xr = np.cos(ang)*x_ - np.sin(ang)*y_
    yr = np.sin(ang)*x_ + np.cos(ang)*y_
    return(xr+180,yr+180)

def search_angle_to_verticalize_dist(x,y,bins=90*4):
    """
   Finds the angle to rotate a distribution of diagonal lines into vertical 
   columns.

   Parameters
   ----------
   x : array-like
       x-coordinates of the data.
   y : array-like
       y-coordinates of the data.
   bins : int, optional
       Number of bins for histogram computation. Default is 360.

   Returns
   -------
   float
       Angle in degrees that verticalizes the distribution.
   """
    maxvert=0
    angc=None
    for ang in np.linspace(0,180,720):
        xr,yr = rot(x,y,ang)
        
        mask = (yr>0) & (yr<=360)
        
        hist,_=np.histogram(yr[mask],bins=bins)
        yhist_disparity = np.std(hist)
        hist,_=np.histogram(xr[mask],bins=bins)
        xhist_disparity = np.max(hist)-np.min(hist)

        verticality=xhist_disparity/yhist_disparity
        
        if verticality>maxvert:
            maxvert=verticality
            angc=ang
    return(angc)

def detect_vertical_peaks(xvert,yvert,bins=90*4,ext=1,verbose=False):
    """
    Detects the locations of vertical peaks in a distribution.

    Parameters
    ----------
    xvert, yvert : array-like
        Arrays representing the data.
    bins : int, optional
        Number of bins for histogram computation. Default is 360.
    ext : int, optional
        Extension factor for the range. Default is 1.
    verbose : bool, optional
        If True, provides additional information. Default is False.

    Returns
    -------
    numpy.ndarray or tuple
        Locations of peaks, or a tuple with peaks and diagnostic information 
        if verbose is True.
    """
    bin2deg=360*(2*ext+1)/bins
    
    mask = (yvert>0) & (yvert<=360) & (xvert>0-ext*360) & (xvert<=360+ext*360)
    x_ytr = xvert[mask]  # truncado en eje y
    
    xhist,_ = np.histogram(x_ytr,bins=bins)
    Nmax=np.max(xhist)
    xhist = xhist/Nmax
    
    peaks, properties = scipy.signal.find_peaks(xhist,prominence=0.5)
    peaks=peaks*bin2deg - ext*360
    
    peaks = peaks[1:-1]
    peak_seps = np.diff(peaks)

    if (np.max(peak_seps)-np.min(peak_seps))/np.min(peak_seps)>1:
        if verbose: return(None,'uneven')
        return(None)
    if peaks[0]>0 or peaks[-1]<=360:
        if verbose: return(None,'few peaks')
        return(None)
    if np.shape(peaks) == (0,): 
        if verbose: return(None,'no peaks')
        return(None)
    
    if verbose: return(peaks,'succesful')
    return(peaks)

def re_rotate_and_count_crossings(vert_peaks,ang,loc=False):
    """
    Computes the crossings of vertical peaks with x and y axes after rotation.

    Parameters
    ----------
    vert_peaks : array-like
        Locations of vertical peaks.
    ang : float
        Rotation angle in degrees.
    loc : bool, optional
        If True, also returns the locations of crossings. Default is False.

    Returns
    -------
    tuple
        Number of x- and y-axis crossings, and optionally their locations.
    """
    if vert_peaks is None: return(0,0,[],[])
    xcross=0
    ycross=0
    xloc=[]
    yloc=[]
    for peaki in vert_peaks:
        p1x,p1y = rot(peaki,0,-ang)
        p2x,p2y = rot(peaki,360,-ang)
        
        xaxval = (p2x - p1x)/(p2y - p1y) * (-p1y) + p1x
        yaxval = (p2y - p1y)/(p2x - p1x) * (-p1x) + p1y
        if (yaxval<360)&(yaxval>=0): 
            ycross+=1
            yloc.append(yaxval)
        if (xaxval<360)&(xaxval>=0): 
            xcross+=1
            xloc.append(xaxval)
    if not loc: return(xcross,ycross)
    return(xcross,ycross,xloc,yloc)

def count_ax_crossings(x,y,loc=False):
    """
    Counts the crossings of strokes with the x- and y- axes.

    Parameters
    ----------
    x, y : array-like
        Data arrays.
    loc : bool, optional
        If True, also returns the locations of crossings. Default is False.

    Returns
    -------
    tuple
        Number of x- and y-axis crossings, and optionally their locations.
    """
    ext = 1    
    bins = 90*4
    xe,ye = extend_angles_2D(x,y,ext=1)
    ang=search_angle_to_verticalize_dist(xe,ye,bins=bins)
    xvert,yvert = rot(xe,ye,ang)
    vert_peaks,flag = detect_vertical_peaks(xvert, yvert,bins=bins,ext=1,verbose=True)
    if flag=='few peaks':  # we extend the limits and increase binsize
        ext = 2    
        bins = 90*6
        xe,ye = extend_angles_2D(x,y,ext=ext)
        ang=search_angle_to_verticalize_dist(xe,ye,bins=bins)
        xvert,yvert = rot(xe,ye,ang)
        vert_peaks = detect_vertical_peaks(xvert, yvert,bins=bins,ext=ext)
    xc,yc,xloc,yloc = re_rotate_and_count_crossings(vert_peaks,ang,loc=True)
    if not loc: return(xc,yc)
    return(xc,yc,xloc,yloc)

def detect_2pmmr(M1,M2,lam1,lam2,verbose=True):
    """
    Detects a 2P-MMR based on orbital parameters. 
    Based on Forgacs-Dajka et al. (2018)

    Parameters
    ----------
    M1, M2 : array-like
        Mean anomalies of the inner and outer bodies, respectively.
    lam1, lam2 : array-like
        Mean longitudes of the inner and outer bodies, respectively.
    verbose : bool, optional
        If True, prints detailed output. Default is True.

    Returns
    -------
    tuple or None
        Resonance (p+q,p) parameters if found, else (0,0).
    """
    # angle with q*vp1
    q,pq = count_ax_crossings(M1, lam2-lam1)
    p = pq - q
    res1 = (pq,p)
    
    # angle with q*vp2
    q,p  = count_ax_crossings(M2, lam2-lam1)
    pq = p + q
    res2 = (pq,p)
    
    if verbose:
        res = res1 if res1!=(0,0) else None
        res1state = 'circ' if res1==(0,0) else 'lib'
        res2state = 'circ' if res2==(0,0) else 'lib'
        print(f'2P-MMR found  -->  {res}')
        print('-----------------------')
        print(f'fi + q*vp1    -->  {res1state}')
        print(f'fi + q*vp2    -->  {res2state}')
        print('-----------------------')
        return
    else:
        return(res1,res2)
    
def plot_forgacs(M1,M2,lam1,lam2,instructions=True,**scatter_kwargs):
    """
    Plots the relevant plots to find the intervening 2-planet mean-motion
    resonance (MMR), as per the process described in Forgacs-Dajka et al. 
    (2018).

    Parameters
    ----------
    M1, M2 : array-like
        Mean anomalies of the inner and outer bodies, respectively.
    lam1, lam2 : array-like
        Mean longitudes of the inner and outer bodies, respectively.
    instructions : bool, optional
        Print instructions explaining how to use the x- and y-crossings to
        calculate the 2-planet MMR. The default is True.
    **scatter_kwargs : dict, optional
        Additional keyword arguments to pass to the scatter plot.

    Returns
    -------
    None.

    """
    
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    
    axs[0].scatter(M1,lam2-lam1,**scatter_kwargs)
    axs[0].set_xlabel(r'$M_1$')
    axs[0].set_ylabel(r'$\lambda_2-\lambda_1$')
    axs[0].set_title(r'$\sigma_1=\varphi(\lambda_1,\lambda_2) + q\ \varpi_1$')
    axs[0].set_xlim(0,360)
    axs[0].set_ylim(0,360)
    
    axs[1].scatter(M1,lam2-lam1,**scatter_kwargs)
    axs[1].set_xlabel(r'$M_1$')
    axs[1].set_ylabel(r'$\lambda_2-\lambda_1$')
    axs[1].set_title(r'$\sigma_2=\varphi(\lambda_1,\lambda_2) + q\ \varpi_2$')
    axs[1].set_xlim(0,360)
    axs[1].set_ylim(0,360)
    plt.show()
    
    print('Instructions for a (p+q)/p mean-motion resonance:')
    print('')
    print('        Resonant angle         |  x-crossings  |  y-crossings')
    print('-------------------------------------------------------------')
    print('p lam1 - (p+q) lam2 + q vp1    |       q       |     p + q')
    print('p lam1 - (p+q) lam2 + q vp2    |       q       |       p')
    return