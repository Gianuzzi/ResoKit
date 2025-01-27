from rsk_io import load_integration
import matplotlib.pyplot as plt
import numpy as np
from rsk_core import Angles
import scipy


def extend_angles(angles, ext):
    """Extends angles from (0, 360) to (0 - ext, 360 + ext)."""
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


plt.close()

sys1 = load_integration(
    "datasets/2planet_example.dat",
    npl=2,
    names=["times", "ibody", "a", "e", "inc", "M", "w", "Omega", "_","resangs"],
)

# sys1 = load_integration(
#     "datasets/nonrescase.dat",
#     npl=2,
#     names=["times", "ibody", "a", "e", "inc", "M", "w", "Omega","resangs"],
# )

# sys1 = load_integration(
#     "datasets/21mmr.dat",
#     npl=2,
#     names=["times", "ibody", "a", "e", "inc", "M", "w", "Omega","resangs"],
# )


pl1 = sys1.planets[0]
pl2 = sys1.planets[1]

lam1 = pl1.lam
lam2 = pl2.lam

M1 = pl1.M
M2 = pl2.M

#-----------

def extend_angles_2D(x,y):
    x0 = np.copy(x)
    y0 = np.copy(y)

    x1 = np.copy(list(x0)*3)
    y1 = np.copy(extend_angles(y0,360))
    
    x = np.copy(extend_angles(x1,360))
    y = np.copy(list(y1)*3)
    return(x,y)

# x,y = extend_angles_2D(M1, lam2-lam1)
x,y = extend_angles_2D(M2, lam2-lam1)


# plt.figure(figsize=(7,7))
# plt.scatter(M1,lam2-lam1,s=.5,zorder=100,c='tab:blue')
# plt.scatter(x1,y1,s=.5,c='tab:red')
# plt.scatter(x2,y2,s=.5,c='tab:green')
# plt.xlim(-360,720)
# plt.ylim(-360,720)
# plt.show() 

def rot(x_,y_,ang):
    x = np.copy(x_)
    y = np.copy(y_)
    ang=ang*np.pi/180
    x-=180
    y-=180
    xn = np.cos(ang)*x - np.sin(ang)*y
    yn = np.sin(ang)*x + np.cos(ang)*y
    return(xn+180,yn+180)

##################################################################
#####        BUSCO EL ANGULO PARA VERTICALIZAR LA DIST       #####
##################################################################
def search_angle_to_verticalize_dist(x,y,bins=90*4,npoints=720):
    maxstd=0
    angc=None
    for ang in np.linspace(0,180,npoints):
        xr,yr = rot(x,y,ang)
        
        mask = (xr>0) & (xr<=360) & (yr>0) & (yr<=360)
        
        hist,_=np.histogram(xr[mask],bins=bins)
        Nmax = len(xr[mask])
        hist = hist/Nmax
        std = np.std(hist)
        
        if std>maxstd:
            maxstd=std
            angc=ang
    return(angc)


################################################
#####              BUSCO PICOS             #####
################################################
def detect_vertical_peaks(xvert,yvert,bins=90*4):
    bin2deg=360*3/bins
    
    mask = (yvert>0) & (yvert<=360) & (xvert>-360) & (xvert<=720)
    x_ytr = xvert[mask]  # truncado en eje y
    y_ytr = yvert[mask]  # truncado en eje y
    
    xhist,_ = np.histogram(x_ytr,bins=bins)
    Nmax=np.max(xhist)
    xhist = xhist/Nmax
    
    peaks, properties = scipy.signal.find_peaks(xhist,prominence=0.5)
    peaks=peaks*bin2deg - 360
    
    peaks = peaks[1:-1]
    peak_seps = np.diff(peaks)

    if (np.max(peak_seps)-np.min(peak_seps))/np.min(peak_seps)>1: return(None)
    if peaks[0]>0 or peaks[-1]<=360: return(None)
    if np.shape(peaks) == (0,): return(None)
    
    return(peaks)

##########################################################
#####    RE-ROTO LOS PICOS Y BUSCO CRUCES EN EJES    #####
##########################################################
def re_rotate_and_count_crosses(vert_peaks,ang):
    if vert_peaks is None: return(0,0)
    xcross=0
    ycross=0
    for peaki in vert_peaks:
        p1x,p1y = rot(peaki,0,-ang)
        p2x,p2y = rot(peaki,360,-ang)
        
        xaxval = (p2x - p1x)/(p2y - p1y) * (-p1y) + p1x
        yaxval = (p2y - p1y)/(p2x - p1x) * (-p1x) + p1y
        if (yaxval<360)&(yaxval>=0): ycross+=1
        if (xaxval<360)&(xaxval>=0): xcross+=1
    return(xcross,ycross)

bins=90*4
ang=search_angle_to_verticalize_dist(x,y,bins=bins)
xvert,yvert = rot(x,y,ang)
vert_peaks = detect_vertical_peaks(xvert, yvert,bins=bins)
xc,yc = re_rotate_and_count_crosses(vert_peaks,ang)

# #### PLOTEAR HISTOGRAMA
# plt.figure()
# plt.plot(np.linspace(-360,720,bins),xhist)
# for peak in peaks:
#     plt.gca().axvline(peak,c='k')
# plt.show()



################################################
#####                 PLOTS                #####
################################################

plt.figure(figsize=(7,7))
 

##### X-AXIS BAND POINTS
# plt.scatter(x_ytr,y_ytr,zorder=140,s=.5,c='red')

plt.scatter(x,y,s=.5,zorder=90,c='k')
# plt.scatter(xvert,yvert,s=.5,zorder=101,c='tab:green')

plt.gca().axhline(0,xmin=0,xmax=360,c='k',zorder=190)
plt.gca().axhline(360,xmin=0,xmax=360,c='k',zorder=190)
plt.gca().axvline(0,ymin=0,ymax=360,c='k',zorder=190)
plt.gca().axvline(360,ymin=0,ymax=360,c='k',zorder=190)


#### PLOT VERTICAL PEAKS
for peaki in vert_peaks:
    p1x,p1y = rot(peaki,0,-0.01)
    p2x,p2y = rot(peaki,360,-0.01)
    
    xlin = np.asarray([-400,760])
    
    ylin = (p2y - p1y)/(p2x - p1x) * (xlin-p1x) + p1y
    plt.plot(xlin,ylin,lw=5,zorder=150,c='yellow')

# #### OVERLAY PEAK HISTOGRAM
# plt.plot(np.linspace(-360,720,bins),xhist*720,c='blue',zorder=150,lw=5)

# #### PLOT DIAGONAL PEAKS AND COUNT CROSSES
# xcross=0
# ycross=0
# for peaki in vert_peaks:
#     p1x,p1y = rot(peaki,0,-ang)
#     p2x,p2y = rot(peaki,360,-ang)
    
#     xlin = np.asarray([-400,760])
    
#     ylin = (p2y - p1y)/(p2x - p1x) * (xlin-p1x) + p1y
    
#     xaxval = (p2x - p1x)/(p2y - p1y) * (-p1y) + p1x
#     yaxval = (p2y - p1y)/(p2x - p1x) * (-p1x) + p1y
#     if (yaxval<360)&(yaxval>=0): ycross+=1
#     if (xaxval<360)&(xaxval>=0): xcross+=1
    
#     plt.plot(xlin,ylin,lw=5,zorder=150,c='yellow')   


plt.xlim(-360,720)
plt.ylim(-360,720)
plt.show() 



