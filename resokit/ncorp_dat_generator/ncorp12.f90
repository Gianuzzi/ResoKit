!!!
!!!   ncorp12.f90 : N-body integrator for planetary systems. (v 20/12/2023)
!!!
!!!   version con estimativa de tiempo de inestabilidad via entropia de Shannon.
!!!
module common
  implicit real*8 (a-h,k-z)
  parameter (imax=4011)
  real*8 twopi,cero,uno,dos,tres2,uno2,pi,G,rad,error,unoc2,Gsum0
  real*8 body0(0:imax),elem(7,imax),eleini(7,imax),eleini0(7,imax)
  real*8 body(0:imax),radius(0:imax),rho(0:imax),prot(0:imax)
  real*8 rhill(0:imax),eta(0:imax),tesc(imax),chaos(4,imax)
  real*8 amin(imax),amax(imax),emin(imax),emax(imax),dimin(imax),dimax(imax)
  real*8 coefc(imax),alfa(imax),egas,wgas0,ggas,rhop,dj2mod,dadty(imax)
  real*8 airy(0:2000),other(7,imax),semi(imax),exc(imax),ene(imax)
  real*8 step,deltat,t0,tstop,time,eps,tout,sgn,unitt,unitm,unitd
  real*8 stepmin,stepmax,facm,dd0,unitmp,ascale,tdump,deltad
  real*8 Q_e,ang_fac,unor2pi,fac_stokes,flare
  real*8 t_stokes,t_disk,sigma0,denpot,Hr0,ric,delta_ic,fac_mig
  real*8 k2delt0(10),k2delti(10),love(0:10),fac_zmi(0:10),zmi(0:10),fac_tid
  real*8 rmin2,rmax2,semimin,semimax,rhmin,demax,chaosmax
  real*8 L2boxmin(imax),L2boxmax(imax),G2boxmin(imax),G2boxmax(imax)
  real*8 dist_box(imax),tesc_S(imax),K_S,qacc_shan,tmin_shan
  integer*4 ll,npl,npart,ntot,neq0,neqs,neqlyp,neq,npl0,npart0,ntot0,iout,iind
  integer*4 inty,iscr,ienel,ietyp,ior,ios,inout,iang(14),inout0,ioout,icout
  integer*4 ifil,idec,idecs,ista,im,ishannon
  integer*4 irel,itid,imag,idrag,irtyp,idu1,iyar,irelp,icaos(16)
  integer*4 icalcm(imax),imig_stokes,imig_typ1,icav
  integer*4 igenout,iplaout,iparout,iencout,ichaout,idmpout
  integer*4 iorder(0:imax),iom,ifstop,ienc_first,name(imax),jr(0:10)
  integer*4 iesc(0:imax),icol(0:2)
  integer*4 iqa_shan,iqe_shan,ifirst
  !
  character(len=90) arch,arch_big,arch_sma,arch_enc,ardp,arch_l
  character(len=90) archp_in,archp_out,arch_body(imax)
  character(len=1)  lectura(300,180)
  save
end module common


program main ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),nrat(imax)
  !      
!!! input data file.
  open (1,file='ncorp12.in',status='old')
  
!!! initialization of global constants.
  twopi = 8.0d0*datan(1.0d0)  ;  cero  = 0.0d0       ;  uno   = 1.0d0
  dos   = 2.0d0               ;  uno2  = uno/dos     ;  tres2 = 3.0d0/2.0d0
  G     = 1.720209895d-02**2  ;  pi    = uno2*twopi  ;  rad   = twopi/360.0d0
  error = 1.0d-13

!!! read data and open output files.
  call data (y)
  
!!!!!! main integration loop
  do while (time < tstop)

!!! call main routine for numerical integration.
    call integrate (y)

!!! calculate mean-motion ratios between adjacent bodies.    
    do i = 2,ntot
      nrat(i) = sqrt((eta(i-1)/eta(i))/(elem(1,i-1)/elem(1,i))**3)
    end do
    nrat(1) = nrat(2)
    
!!! if requested, continue calculation of Shannon entropy.
!    if (ishannon > 0) call shannon (y)

!!! output of results.
    do i = 1,ntot
      j = name(i)
      if (igenout == 1) then                 ! general output file
        write (3,100) tout,j,elem(1:inout,i),other(1:ioout,i),chaos(1:icout,j)
        !,nrat(i)!,tesc_S(i)
        flush (3)
      end if
      if (iscr == 1) then                    ! output on screen
        write (*,100) tout,j,elem(1:inout,i),other(1:ioout,i),chaos(1:icout,j)
      end if
      if (iind == 1) then                    ! individual file per body
       write (100+j,99) tout,elem(1:inout,i),other(1:ioout,i),chaos(1:icout,j) 
        flush (100+j)
      end if
      if (i <= npl .and. iplaout == 1) then  ! planetary output file
       write (11,100) tout,j,elem(1:inout,i),other(1:ioout,i),chaos(1:icout,j)
        flush (11)
      end if
      if (i > npl .and. iparout == 1) then   ! particle output file
        write (12,100) tout,j,elem(1:inout,i),other(1:ioout,i),chaos(1:icout,j)
        flush (12)
      end if
    end do

!!! if requested, show percentage of total integration time.
    if (iscr == -1) call percentage (time,tstop)

!!! update file of chaos indicators.
    if (icout > 0 .and. ichaout == 1) then
      open (13,file=arch_l,status='replace')
      do i = 1,ntot0
        write (13,100) tout,i,eleini0(1:6,i),chaos(1:icout,i)!,tesc_S(i)
      end do
      close (13)
    end if
    
!!! if lyapunov turned on, renormalize variational vector.
    if (maxval(icaos(1:6)) > 0) call normalize_lyap (y)
    
!!! if scheduled, do periodic dump.
    if (idmpout == 1 .and. tout >= tdump) then
      call dump (1,y) ; tdump = tdump + deltad
    end if
    
!!! in a run with particles, stop all massless bodies were ejected.
    if (npart0 > 0 .and. npart == 0) goto 81
    
!!! if requested, stop run after a planet is ejected or suffers collision.
    if (ifstop > 0 .and. npl /= npl0) goto 81
       
!!!!! end of integration loop.
  end do
81 continue

!!! if requested, write on screen final values of chaos indicators.
  if (ichaout == -1) then
    if (npl0 == ntot0) then  ! no particles present in simulation
      do i = 1,ntot0
        write (*,120) tout,i,body0(i),eleini0(1:6,i),chaos(1:icout,i)!,tesc_S(i)
      end do
    else
      do i = npl0+1,ntot0     ! only include particles in output
        write (*,110) tout,i,eleini0(1:6,i),chaos(1:icout,i)!,tesc_S(i)
      end do
    end if
  end if
  !
99  format (1pe15.7,14e18.9)
100 format (1pe15.7,i6,14e16.7)
110 format (1pe15.7,i6,0p6f16.9,1p14e16.7)
120 format (1pe15.7,i6,e16.7,0p6f16.9,1p14e16.7)
  !
end program main
      
      
subroutine data (y) ; use common
  implicit real*8 (a-h,k-z)
  integer*4 ncarteles,jbeg(300)
  real*8 y(18*imax),y0(18*imax),yy(6),dist(imax),qtid(0:10)
  real*8 inc(imax),anom(imax),w(imax),om(imax)
  character(len=200) dummy
  character(len=90)  linres,rdata,rtype,integ
  character(len=90)  bunit,bunit2,unit,eletype,ask_mig_stokes,ask_mig_typ1
  character(len=90)  ask_tid,ask_pla,bunitp,ask_rel,ask_par,ask_sto,ask_yar
  character(len=90)  ask_relp,ask_cav,ask_lyap,ask_ifs,ask_fil,ask_mag,outind
  character(len=90)  outscr,outtype,outrad,outsp,outel,eletypeo,outmas
  character(len=18)  cartel(86)
  character(len=5)   ajr(10),arho(10),fin
  character(len=1)   letras_res(50),letras_lyap(16)
  logical mask(imax)
  !
!!! some additional parameters and/or constants.     
  zmsolkg = 1.98911d30               ! solar mass in kg
  diaseg  = 60.0*60.0*24.0           ! day in seconds
  uam     = 1.495978707d11           ! AU in meters
  vluz    = 2.99792458d8*diaseg/uam  ! light speed [AU/day]
  unoc2   = uno/vluz**2              ! inverse light speed squared
  dd0     = 1.0d-06                  ! initial separation for LCE
  facm    = 1.0d4                    ! ad-dedoc scaling factor for Megno
  unor2pi = uno/sqrt(twopi)          ! factor useful for gas density
  
!!! initialization of variables.
  body0  = cero  ;  body   = cero   ;  y0      = cero  ;   y      = cero
  coefc  = cero  ;  alfa   = cero   ;  radius  = cero  ;   rho    = uno
  rhill  = cero  ;  zmi    = cero   ;  love    = cero  ;   dadty  = cero
  eta    = cero  ;  emin   = uno    ;  emax    = cero  ;   dimin  = twopi
  dimax  = cero  ;  amin   = 1.d15  ;  amax    = cero  ;   prot   = 1.0
  tesc   = 1.d10 ;  icalcm = 0      ;  iesc(0) = 0     ;  icol(0) = 0
  ienc_first = 0 ; fac_stokes = uno ; fac_mig  = uno   ;       jr = 0
  ifirst = 0
  
!!! load code's vocabulary.
  call vocabulary (cartel,ncarteles)
  
!!! read input data file.
  iflag = 0
  i = 1
  do while (iflag == 0) 
    read (1,'(a)') dummy
    read(dummy,'(180a1)') (lectura(i,j),j=1,180)
    close (55)
!!! look for first significant character in each line.
    jbeg(i) = 0
    j = 1
    do while (jbeg(i) == 0 .and. j < 180)
      if (lectura(i,j) /= '') jbeg(i) = j
      j = j + 1
    end do
    if (jbeg(i) > 80) jbeg(i) = 0
    do j = 1,78
      fin = lectura(i,j)//lectura(i,j+1)//lectura(i,j+2)
      if (fin(1:3) == 'end') goto 1
    end do
    i = i + 1
  end do
1 ilmax = i - 1             ! finish reading input file
  
!!! type of run (new or restart).
  rtype = 'n' ; irtyp = 0 ; idu1 = 0 ; ardp = 'no' ; deltad = 1.d5
  call option_char (ilmax,jbeg,cartel(1),rtype)
  call option_char (ilmax,jbeg,cartel(2),ardp)
  idmpout = 1 ; if (ardp == 'no') idmpout = 0
  call option_real (ilmax,jbeg,cartel(3),deltad)  
  if (rtype(1:1) == 'r') then ; irtyp = 1 ; idu1 = 1 ; end if
  tdump = deltad

!!! if restart, read dump file, open output files, direct pointer and return.
  if (idmpout == 1 .and. irtyp == 1) then
    call dump (-1,y)
    if (igenout == 1) open (3,file=arch,status='unknown',access='append')
    if (iencout == 1) then
      open (16,file=arch_enc,status='unknown',access='append')
      tcheck = 2.0*tdump
      do while (tcheck > tdump)
        backspace (3) ; read (3,*) tcheck,i ; backspace (3)
      end do
      read (3,*) tcheck,i
    end if
    if (iplaout == 1) then
      open (11,file=arch_big,status='unknown',access='append')
      tcheck = 2.0*tdump
      do while (tcheck > tdump)
        backspace (11) ; read (11,*) tcheck,i ; backspace (11)
      end do
      read (11,*) tcheck,i
    end if
    if (npart > 0 .and. iparout == 1) then
      open (12,file=arch_sma,status='unknown',access='append')
      tcheck = 2.0*tdump
      do while (tcheck > tdump)
        backspace (11) ; read (11,*) tcheck,i ; backspace (11)
      end do
      read (11,*) tcheck,i
    end if
    if (iind == 1) then
      do i = 1,ntot
        open (100+i,file=arch_body(i),status='unknown',access='append')
        tcheck = 2.0*tdump
        do while (tcheck > tdump)
          backspace (100+i) ; read (100+i,*) tcheck ; backspace (100+i)
        end do
        read (100+i,*) tcheck
      end do
    end if
    tdump = tdump + deltad
    return
  end if

!!! basic units.
  bunit = 's' ; bunit2 = 'a' ; unit = 'y'
  call option_char (ilmax,jbeg,cartel(4),bunit)
  call option_char (ilmax,jbeg,cartel(5),bunit2)
  call option_char (ilmax,jbeg,cartel(6),unit)
  if (bunit(1:1) == 's') unitm = 1.0d0
  if (bunit(1:1) == 'j') unitm = 9.54792d-4
  if (bunit(1:1) == 'e') unitm = 3.04043d-6
  if (bunit(1:1) == 'k') unitm = 1.0d0/1.9891d30
  unitd = 1.0d0
  if (bunit2(1:1) == 'k') unitd = 1.495978707d08
  if (bunit2(1:1) == 'm') unitd = 1.495978707d11
  unitt = 1.0d0
  if (unit(1:1) == 's') unitt = 1.0d0/60.0d0/60.0d0/24.0d0
  if (unit(1:1) == 'y') unitt = 365.25635d0
  
!!! parameters of the integration.
  integ = 'bs' ; t0  = cero ; tstop = 1.0d6 ; deltat = 1.0d3 ; ll = 11
  step0 = -1   ; sgn = 1.0
  call option_char (ilmax,jbeg,cartel(7),integ)
  call option_real (ilmax,jbeg,cartel(8),t0)
  call option_real (ilmax,jbeg,cartel(9),tstop)
  call option_real (ilmax,jbeg,cartel(10),deltat)
  call option_int  (ilmax,jbeg,cartel(11),ll)
  call option_real (ilmax,jbeg,cartel(12),step0)
  call option_real (ilmax,jbeg,cartel(13),sgn)
  if (integ(1:2) == 'ra') inty = 1
  if (integ(1:2) == 'bs') inty = 2
  if (integ(1:2) == 'rk') inty = 3
  t0     = t0*unitt
  tstop  = tstop*unitt
  deltat = deltat*unitt
  step0  = step0*unitt
  eps    = 10.0**(-ll)
  if (t0 < error) t0 = error
  
!!! data of the primary.
  body(0) = uno ; radius(0) = 590000.0 ; prot(0) = 27.8 ; dj2mod = 0.0
  call option_real (ilmax,jbeg,cartel(14),body(0))
  call option_real (ilmax,jbeg,cartel(15),radius(0))
  call option_real (ilmax,jbeg,cartel(16),prot(0))
  call option_real (ilmax,jbeg,cartel(76),dj2mod)
  body(0)   = body(0)*unitm                 ! mass in M_sun
  radius(0) = radius(0)*1.0d3/uam           ! solar radius in AU
  Gsum0     = G*body(0)
  
!!! the planets.
  ask_pla = 'y' ; bunitp = 's' ; eletype = 'a'
  call option_char (ilmax,jbeg,cartel(18),ask_pla)
  call option_char (ilmax,jbeg,cartel(17),bunitp)
  call option_char (ilmax,jbeg,cartel(19),eletype)
  if (bunitp(1:1) == 's') unitmp = 1.0d0
  if (bunitp(1:1) == 'j') unitmp = 9.54792d-4
  if (bunitp(1:1) == 'e') unitmp = 3.04043d-6
  if (bunitp(1:1) == 'k') unitmp = 1.0d0/1.9891d30
  ietyp_i = 1
  if (eletype(1:1) == 'b') ietyp_i = 2
  if (eletype(1:1) == 'j') ietyp_i = 3
  if (eletype(1:1) == 'p') ietyp_i = 4
  if (eletype(1:1) == 'm') ietyp_i = 5
  
!!! before analyzing planetary data, checks if Stokes-type migration is on.
  ask_mig_stokes = 'n' ; imig_stokes = 0 ; t_stokes = 1.0d6
  call option_char (ilmax,jbeg,cartel(21),ask_mig_stokes)
  call option_real (ilmax,jbeg,cartel(22),t_stokes)
  if (ask_mig_stokes(1:1).eq.'y') imig_stokes = 1
  t_stokes = t_stokes*unitt
  
!!! reads planetary data: elem(:,i) = (a,e,I,anom,argper,lnode) of body(i).
  npl = 0
  if (ask_pla(1:1) == 'y') then
    do i = 1,ilmax+1
      backspace (1)
    end do
    i = 1 ; j = 1
21  continue
    if (imig_stokes == 0) then
      read (1,*,err=22,end=23) body(i),semi(i),exc(i),inc(i),anom(i),w(i),om(i)
    else
      read (1,*,err=22,end=23) body(i),semi(i),exc(i),inc(i),anom(i),w(i), &
           om(i),tau_a,tau_e
      if (max(abs(tau_a),abs(tau_e)) > 1.0d-13) then
        if (abs(tau_a) < 1.0d-13) tau_a = 1.0d15
        if (abs(tau_e) < 1.0d-13) tau_e = 1.0d15
        coefc(i) = uno2/tau_a + uno/tau_e ; alfa(i) = cero
        if (coefc(i) /= cero) alfa(i) = uno/tau_e/coefc(i)
        coefc(i) = coefc(i)/unitt
      end if
    end if
    semi(i) = semi(i)/unitd
    body(i) = body(i)*unitmp ! transform masses to [Msol]
    i = i + 1
    goto 21
22  j = j + 1
    if (j > ilmax) goto 23
    goto 21
23  continue
!!! total number of massive bodies (not counting primary).
    npl = i - 1
  end if
  
!!! if planets requested, but none found in input file, read from screen.
  if (ask_pla(1:1) == 'y' .and. npl == 0) then
    i = 1 ; j = 1
31  continue
    if (imig_stokes == 0) then
      read (*,*,err=32,end=33) body(i),semi(i),exc(i),inc(i),anom(i),w(i),om(i)
    else
      read (*,*,err=32,end=33) body(i),semi(i),exc(i),inc(i),anom(i),w(i), &
           om(i),tau_a,tau_e
      if (max(abs(tau_a),abs(tau_e)) > 1.0d-13) then
        coefc(i) = uno2/tau_a + uno/tau_e ; alfa(i) = cero
        if (coefc(i) /= cero) alfa(i) = uno/tau_e/coefc(i)
        coefc(i) = coefc(i)/unitt
      end if
    end if
    semi(i) = semi(i)/unitd
    body(i) = body(i)*unitmp ! transform to correct mass units
    i = i + 1
    goto 31
32  j = j + 1
    if (j > ilmax) goto 33
    goto 31
33  continue
    npl = i - 1
  end if
  
!!! indices for mean-motion resonance among planets (if requested).
  rdata = "" ; call option_char (ilmax,jbeg,cartel(77),rdata)
  if (rdata /= "") read (rdata,*) (jr(i),i=1,npl)
  jr(0) = maxval(abs(jr),npl)
  
!!! current count of orbiting bodies.
  ntot = npl
  
!!! type-I migration for planets.
  ask_mig_typ1 = 'n' ; Q_e = 0.1 ; ang_fac = 0.3
  call option_char (ilmax,jbeg,cartel(23),ask_mig_typ1)
  imig_typ1 = 0 ; if (ask_mig_typ1(1:1).eq.'y') imig_typ1 = 1
  call option_real (ilmax,jbeg,cartel(78),Q_e)
  call option_real (ilmax,jbeg,cartel(79),ang_fac)
  
!!! tidal effects for planets.
  ask_tid = 'n' ; qtid = 1.0d6 ; love = 0.3 ; fac_zmi = 0.25 ; fac_tid = 1.0
  call option_char (ilmax,jbeg,cartel(31),ask_tid)
  itid = 0 ; if (ask_tid(1:1) == 'y') itid = 1
  
!!! if requested, read stellar & planetary data necessary for tidal effects.
  if (itid == 1) then
    call option_real (ilmax,jbeg,cartel(34),love(0))
    call option_real (ilmax,jbeg,cartel(35),fac_zmi(0))
    call option_real (ilmax,jbeg,cartel(81),qtid(0))
    rdata = "" ; call option_char (ilmax,jbeg,cartel(82),rdata)
    if (rdata /= "") read (rdata,*) (love(i),i=1,npl)
    rdata = "" ; call option_char (ilmax,jbeg,cartel(83),rdata)
    if (rdata /= "") read (rdata,*) (fac_zmi(i),i=1,npl)
    rdata = "" ; call option_char (ilmax,jbeg,cartel(84),rdata)
    if (rdata /= "") read (rdata,*) (qtid(i),i=1,npl)
    rdata = "" ; call option_char (ilmax,jbeg,cartel(32),rdata)
    if (rdata /= "")  read (rdata,*) (rho(i),i=1,npl)
    rdata = "" ; call option_char (ilmax,jbeg,cartel(33),rdata)
    if (rdata /= "") read (rdata,*) (prot(i),i=1,npl)
    call option_char (ilmax,jbeg,cartel(86),ask_mag)
    imag = 0 ; if (ask_mag(1:1) == 'y') imag = 1
    call option_real (ilmax,jbeg,cartel(85),fac_tid)
  end if

!!! CTL (Mignard 1979) tidal parameter k2*deltat.
  do i = 1,npl
    enei = sqrt(G*(body(0)+body(i))/semi(i)**3)
    k2delt0(i) = 1.5*G*(body(0)+body(i))/qtid(0)/enei
    k2delti(i) = 1.5*G*(body(0)+body(i))/qtid(i)/enei
  end do
  
!!! calculate stellar & planetary radii and moment of inertia.
  zmi(0) = fac_zmi(0)*body(0)*(radius(0)**2)
  do i = 1,npl
    radius(i) = (1.5*body(i)/(rho(i)/1.0d3/zmsolkg)/twopi)**0.333333
    radius(i) = radius(i)/1.0d2/uam   ! cm -> AU
    zmi(i)    = fac_zmi(i)*body(i)*(radius(i)**2)
  end do
  
!!! relativistic effect for planets,
  ask_rel = 'n'
  call option_char (ilmax,jbeg,cartel(36),ask_rel)
  irel = 0 ; if (ask_rel(1:1).eq.'y') irel = 1
  
!!! the massless particles.
  ask_par  = 'n' ; archp_in = 'particles.in' ; ask_sto  = 'n' ; ask_yar  = 'n'
  ask_relp = 'n' ; rhop     = 3.0 ; albedo = 0.04 ; npart = 0
  call option_char (ilmax,jbeg,cartel(37),ask_par)
  call option_char (ilmax,jbeg,cartel(38),archp_in)
  call option_char (ilmax,jbeg,cartel(40),ask_sto)
  call option_real (ilmax,jbeg,cartel(41),rhop)
  rhop = rhop*(1.495978707d13**3)/1.98911d33 ! [Msol/UA^3]
  call option_char (ilmax,jbeg,cartel(48),ask_yar)
  call option_real (ilmax,jbeg,cartel(49),albedo)
  call option_char (ilmax,jbeg,cartel(50),ask_relp)
  idrag = 0 ; if (ask_sto(1:1).eq.'y')  idrag = 1
  iyar  = 0 ; if (ask_yar(1:1).eq.'y')  iyar  = 1
  irelp = 0 ; if (ask_relp(1:1).eq.'y') irelp = 1
  
!!! read particle data from external file.
  if (ask_par(1:1) == 'y' .and. archp_in(1:1) /= '*') then
    open (2,file=archp_in,status='old')
    i = npl + 1
3   continue
    if (idrag+iyar == 0) then 
      read (2,*,end=4) semi(i),exc(i),inc(i),anom(i),w(i),om(i)
    else
      read (2,*,end=4) semi(i),exc(i),inc(i),anom(i),w(i),om(i),radius(i)
      dadty(i)  = albedo*radius(i)  ! in [AU/d]
      radius(i) = radius(i)/uam     ! particle radius in AU
      coefc(i)  = cero
      if (radius(i) > 1.0d-18) coefc(i) = (3.0/8.0)*0.44/radius(i)/rhop
    end if
    semi(i) = semi(i)/unitd
    body(i) = cero
    i = i + 1
    goto 3
4   npart = i - npl - 1
    close (2)
    if (idrag+iyar > 0 .and. radius(npl+1) == 0.0) then
      write (*,*)
      write (*,*) 'Error in particles.in: Particle radii not specified'
      write (*,*)
      stop
    end if
  end if
  
!!! read particle data from screeen (when using ncorp for sectors of a grid).
  if (ask_par(1:1) == 'y' .and. archp_in(1:1) == '*') then
    i = npl + 1
13  continue
    if (idrag+iyar.eq.0) then 
      read (*,*,end=14) semi(i),exc(i),inc(i),anom(i),w(i),om(i)
    else
      read (*,*,end=14) semi(i),exc(i),inc(i),anom(i),w(i),om(i),radius(i)
      dadty(i)  = albedo*radius(i)  ! in [AU/d]
      radius(i) = radius(i)/uam     ! particle radius in AU
      coefc(i)  = cero
      if (radius(i) > 1.0d-18) coefc(i) = (3.0/8.0)*0.44/radius(i)/rhop
    end if
    semi(i) = semi(i)/unitd
    body(i) = cero
    i = i + 1
    goto 13
14  npart = i - npl - 1
    if (idrag+iyar > 0 .and. radius(npl+1) == 0.0) then
      write (*,*)
      write (*,*) 'Error in particles.in: Particle radii not specified'
      write (*,*)
      stop
    end if
  end if
  
!!! total number of bodies (planets + particles).
  ntot = npl + npart
  
!!! in case Jacobi frame is used, order bodies according to semimajor axis.
  dist = 1.0d30
  dist(1:ntot) = semi(1:ntot)               ! semimajor axis
  mask(1:ntot) = .true.
  iorder(0) = 0
  do i = 1,ntot
    iorder(i) = minloc(dist,1,mask)
    mask(iorder(i)) = .false.
  end do
  ! builds mass factors used for conversion to and from Jacobi frame.
  eta(0) = body(0)
  do i = 1,ntot
    eta(iorder(i)) = eta(iorder(i-1)) + body(iorder(i))
  end do
  
!!! transform planetary orbital elements to cartesian coordinates.
  do i = 1,ntot
    if (ietyp_i == 1) bodyc = body(0) + body(i)
    if (ietyp_i == 2) bodyc = (body(0)**3)/((body(0) + body(i))**2)
    if (ietyp_i == 3) bodyc = eta(i)
    if (ietyp_i == 4) bodyc = body(0) + body(i)
    if (ietyp_i == 5) then
      if (i == 1) bodyc = body(0) + body(i)
      if (i > 1)  bodyc = body(0) + body(1) + body(i)
    end if
    ene(i) = sqrt(G*bodyc/semi(i)**3)  ! calculate initial mean motions
    call coord (bodyc,semi(i),exc(i),inc(i),anom(i),w(i),om(i),yy)
    do j = 1,3
      y0(j+6*(i-1))   = yy(j)
      y0(j+6*(i-1)+3) = yy(j+3)*sgn
    end do
    rhill(i) = semi(i)*((0.333333*body(i)/body(0))**0.333333)
  end do
  
!!! change coordinates & velocities to astrocentric reference frame.
  if (ietyp_i == 1) y = y0
  if (ietyp_i == 2) call coord_bh (1,y0,y)
  if (ietyp_i == 3) call coord_jh (1,y0,y)
  if (ietyp_i == 4) call coord_ph (1,y0,y)
  if (ietyp_i == 5) call coord_mh (1,y0,y)
  
!!! the gas disk.
  sigma0 = 1.0d5 ; denpot = 1.0 ; Hr0  = 0.07 ; flare   = 0.0 ; t_disk = 1.0d6
  egas   = 0.0   ; wgas0  = 0.0 ; ggas = 0.0  ; ask_cav = 'n' ; ric = 0.1
  delta_ic = 0.01
  call option_real (ilmax,jbeg,cartel(24),sigma0)
  sigma0 = sigma0*(1.495978707d13**2)/1.98911d33   ! [Msol/UA^2]
  call option_real (ilmax,jbeg,cartel(25),denpot)
  call option_real (ilmax,jbeg,cartel(80),flare)
  call option_real (ilmax,jbeg,cartel(26),Hr0)
  call option_real (ilmax,jbeg,cartel(27),t_disk)
  t_disk = t_disk*unitt
  call option_real (ilmax,jbeg,cartel(45),egas)
  call option_real (ilmax,jbeg,cartel(46),wgas0)
  call option_real (ilmax,jbeg,cartel(47),ggas)
  ggas = ggas*(twopi/365.25)
  call option_char (ilmax,jbeg,cartel(28),ask_cav)
  icav = 0 ; if (ask_cav(1:1).eq.'y') icav = 1
  call option_real (ilmax,jbeg,cartel(29),ric)
  call option_real (ilmax,jbeg,cartel(30),delta_ic)
!!! set gas to Keplerian velocity ratio.
  if (ask_par(1:1).eq.'y') then
    do i = npl+1,ntot
      alfa(i) = uno - Hr0*Hr0
    end do
  end if
  
!!! chaos indicators.
  ask_lyap = 'no' ; ichaos = 0 ; icout = 0 ; chaosmax = 10.0 ; ishannon = 0
  call option_char (ilmax,jbeg,cartel(20),ask_lyap)
  call option_real (ilmax,jbeg,cartel(71),chaosmax)
  do i = 1,16
    letras_lyap(i) = ask_lyap(i:i)
  end do
  if (letras_lyap(1) /= 'n') then
    j = 0
    do i = 1,16
      if (letras_lyap(i) /= ' ') then
        j = j + 1
        if (iachar(letras_lyap(i)) == 108) icaos(j) =  1    ! LCE
        if (iachar(letras_lyap(i)) == 109) icaos(j) =  2    ! <Y>
        if (iachar(letras_lyap(i)) == 101) icaos(j) = -1    ! delta_e
        if (iachar(letras_lyap(i)) == 97)  icaos(j) = -2    ! delta_a
        if (iachar(letras_lyap(i)) == 105) icaos(j) = -3    ! delta_i
        if (iachar(letras_lyap(i)) == 116) icaos(j) = -4    ! t_escape
        if (iachar(letras_lyap(i)) == 115) then
          ishannon = 1         ;           icaos(j) = -5    ! Shannon t_esc
        end if
      end if
    end do
  end if
  do i = 1,16
    if (icaos(i) /= 0) icout = icout + 1
  end do

!!! if Shannon entropy requested, initialize necessary quantities.
  if (ishannon > 0) then
    call per_sec (tau_sec)
    call init_shannon (tau_sec)
  end if
  
!!! conditions for escape.
  rmin2 = 0.001**2 ; rmax2 = 100.0**2 ; semimin = 0.001 ; semimax = 100.0
  rhmin = 0.1      ; demax = 0.9      ; ifstop  = 0
  call option_real (ilmax,jbeg,cartel(51),rmin)
  call option_real (ilmax,jbeg,cartel(52),rmax)
  call option_real (ilmax,jbeg,cartel(53),rhmin)
  call option_real (ilmax,jbeg,cartel(72),semimin)
  call option_real (ilmax,jbeg,cartel(73),semimax)
  call option_real (ilmax,jbeg,cartel(54),demax)
  call option_char (ilmax,jbeg,cartel(75),ask_ifs)
  if (ask_ifs(1:1).eq.'y') ifstop = 1
  rmin2 = rmin*rmin ; rmax2 = rmax*rmax
  
!!! filter stuff.
  ask_fil = 'n' ; idec = 1 ; im = 200 ; idecs = 1
  call option_char (ilmax,jbeg,cartel(55),ask_fil)
  call option_int  (ilmax,jbeg,cartel(56),idec)
  call option_int  (ilmax,jbeg,cartel(57),im)
  call option_int  (ilmax,jbeg,cartel(58),idecs)
  ifil = 0
  if (ask_fil(1:1).eq.'y') then
    ifil = 1 ; call filter 
  end if
  
!!! ouput files and type.
  arch     = 'ncorp11.dat' 
  arch_big = 'planets.dat' ; arch_sma = 'particles.dat'
  arch_l   = 'chaos.dat'   ; arch_enc = 'encounters.dat'
  outind   = 'n' ; outscr = 'y' ; outtype  = 'e' ; eletypeo = 'a' 
  outmas   = 'n' ; outrad = 'n' ; outsp    = 'n' ; outel    = 'n'
  igenout  = 1 ; iplaout  = 1 ; iparout  = 1 ; iencout  = 1 ; ichaout  = 1
  call option_char (ilmax,jbeg,cartel(59),arch)
  if (arch.eq.'no') igenout = 0
  call option_char (ilmax,jbeg,cartel(60),arch_big)
  if (arch_big.eq.'no') iplaout = 0
  call option_char (ilmax,jbeg,cartel(61),arch_sma)
  if (arch_sma.eq.'no') iparout = 0
  call option_char (ilmax,jbeg,cartel(74),arch_enc)
  if (arch_enc.eq.'no') iencout = 0
  call option_char (ilmax,jbeg,cartel(62),arch_l)
  if (arch_l.eq.'no') ichaout =  0
  if (arch_l.eq.'*')  ichaout = -1
  call option_char (ilmax,jbeg,cartel(63),outind)
  call option_char (ilmax,jbeg,cartel(64),outscr)
  call option_char (ilmax,jbeg,cartel(65),outtype)
  call option_char (ilmax,jbeg,cartel(66),eletypeo)
  call option_char (ilmax,jbeg,cartel(67),outmas)
  call option_char (ilmax,jbeg,cartel(68),outrad)
  call option_char (ilmax,jbeg,cartel(69),outsp)
  call option_char (ilmax,jbeg,cartel(70),outel)
  iind = 0  ; if (outind(1:1)  == 'y') iind  = 1
  iscr = 0  ; if (outscr(1:1)  == 'y') iscr  = 1 ; if (outscr == '%') iscr = -1
  iout = 0  ; if (outtype(1:1) == 'c') iout  = 1
  iom  = 0  ; if (outmas(1:1)  == 'y') iom   = 1
  ior  = 0  ; if (outrad(1:1)  == 'y') ior   = 1
  ios  = 0  ; if (outsp(1:1)   == 'y') ios   = 1
  ienel = 0 ; if (outel(1:1)   == 'y') ienel = 1
  ietyp = 1
  if (eletypeo(1:1) == 'b') ietyp = 2
  if (eletypeo(1:1) == 'j') ietyp = 3
  if (eletypeo(1:1) == 'p') ietyp = 4
  if (eletypeo(1:1) == 'm') ietyp = 5
  
!!! if requested, give each planet its own individual output file.
  if (iind.eq.1) then
    do i = 1,npl
      write (rdata,'(i0)') i ; arch_body(i) = 'planet'//trim(rdata)//'.dat'
    end do
    do i = npl+1,ntot
      write (rdata,'(i0)') i-npl ; arch_body(i)='particle'//trim(rdata)//'.dat'
    end do
  end if
  
!!! initialize system scale as minimum semmajor axis.
  ascale = semi(1) ; do i = 2,ntot ; ascale = min(ascale,semi(i)) ; end do
  
!!! initialize integration step as fraction of orbital period of inner body.
  tau  = twopi*sqrt((ascale**3)/Gsum0)
  step = max(step0,0.005*tau)
  stepmin = 1.0d-10*tau     ! minimum possible step size (exagerated)
  stepmax = 0.05*tau        ! maximum possible step size (exagerated)
  
!!! initialize name list.
  do i = 1,ntot ; name(i) = i ; end do
  
!!! current total number of differential equations (orbital N-body problem).
  neq0 = 6*ntot
  neq  = neq0
  
!!! number of orbital quantities per output line.
  inout = 6
  if (jr(0) /= 0) inout = inout + 1    ! write resonant angle
  
!!! number of additional parameters per output line.
  ioout = 0
  if (iom /= 0)   ioout = ioout + 1    ! write planetary mass
  if (ior /= 0)   ioout = ioout + 1    ! write planetary radius
  if (ios /= 0)   ioout = ioout + 3    ! write planetary/stellar spin
  if (ienel == 1) ioout = ioout + 2    ! write energy and ang. momentum
  
!!! save initial masses and orbital elements for chaos indicator maps.
  iang = 0 ; inout0 = 6 ; if (jr(0) /= 0) inout0 = 7
  do i = 1,ntot
    eleini(1,i) = semi(i)
    eleini(2,i) = exc(i)
    eleini(3,i) = inc(i)
    eleini(4,i) = anom(i)
    eleini(5,i) = w(i)
    eleini(6,i) = om(i)
    if (jr(0) /= 0) eleini(7,i) = cero
  end do
  iang(4) = 1 ; iang(5) = 1 ; iang(6) = 1 ; if (jr(0) /= 0) iang(7) = 1
  body0 = body
  
!!! number of EDOs & initial conditions for spins.
  neqs = 0
  if (itid == 1 .or. ios == 1) then
    neqs = 3*(npl+1)           ! number of spin equations
    y(neq0+1) = cero
    y(neq0+2) = cero
    y(neq0+3) = twopi/prot(0)  ! initially perpendicular to reference plane
    do i = 1,npl
      y(neq0+3*i+1) = cero
      y(neq0+3*i+2) = cero
      y(neq0+3*i+3) = twopi/prot(i)
    end do
  end if
  
!!! update total number of differential equations.
  neq = neq + neqs
  
!!! initial conditions for orbital variational equations (Lyapunov).
  neqlyp = 0
  if (maxval(icaos(1:6)) > 0) then
    neqlyp = 9*ntot
    do i = 1,ntot
      y(neq+9*(i-1)+1) = dd0
      y(neq+9*(i-1)+2) = dd0
      y(neq+9*(i-1)+3) = dd0
      y(neq+9*(i-1)+4) = dd0
      y(neq+9*(i-1)+5) = dd0
      y(neq+9*(i-1)+6) = dd0
      y(neq+9*(i-1)+7) = 1.0d-12 ! initial conditions for Lyapunov exponent
      y(neq+9*(i-1)+8) = 1.0d-12 ! idem for Megno
      y(neq+9*(i-1)+9) = 1.0d-12 ! idem for averaged Megno
    end do
  end if
  
!!! update total number of differential equations.
  neq = neq + neqlyp
  
!!! save body numbers and initial conditions.
  npl0 = npl ; npart0 = npart ; ntot0 = ntot
  eleini0(:,1:ntot) = eleini(:,1:ntot)
  
!!! open output files.
  if (igenout == 1) open (3,file=arch,status='replace')
  if (iplaout == 1) open (11,file=arch_big,status='replace')
  if (npart > 0 .and. iparout == 1) open (12,file=arch_sma,status='replace')
  if (iind == 1) then
    do i = 1,ntot
      open (100+i,file=arch_body(i),status='replace')
    end do
  end if
!
end subroutine data


subroutine output (y,orele) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),y0(18*imax),orele(7,imax),r2(imax),x(6)
  !
  orele(:,1:ntot) = cero
  other(:,1:ntot) = cero
  
!!! calculate total orbital energy and angular momentum.
  if (ienel == 1) call energia (y,energ,cang)
  
!!! change to chosen type of coordinates.
  if (ietyp == 1) y0 = y
  if (ietyp == 2) call coord_bh (-1,y0,y)
  if (ietyp == 3) call coord_jh (-1,y0,y)
  if (ietyp == 4) call coord_ph (-1,y0,y)
  if (ietyp == 5) call coord_mh (-1,y0,y)
  
!!! if backward integration, return to normal velocities.
  if (sgn < cero) then
    do i = 1,ntot
      y0(6*i-2) = -y0(6*i-2)
      y0(6*i-1) = -y0(6*i-1)
      y0(6*i)   = -y0(6*i)
    end do
  end if
  
!!! calculate output variables. 
  iah = 6
  do i = 1,ntot
    x = y0(1+6*(i-1):6*i)        
    if (ietyp == 1) body_sum = body(0) + body(i)
    if (ietyp == 2) body_sum = (body(0)**3)/((body(0) + body(i))**2)
    if (ietyp == 3) body_sum = eta(i)
    if (ietyp == 4) body_sum = body(0) + body(i)
    if (ietyp == 5) then
      if (i == 1) body_sum = body(0) + body(i)
      if (i > 1)  body_sum = body(0) + body(1) + body(i)
    end if
    call elements (body_sum,x,semi(i),exc(i),di,dm,wi,omi)
    ene(i)   = sqrt(G*body_sum/semi(i)**3)
    ascale   = min(ascale,semi(i))
    amin(i)  = min(amin(i),semi(i)*unitd) ! update min(a)
    amax(i)  = max(amax(i),semi(i)*unitd) ! update max(a)
    emin(i)  = min(emin(i),exc(i))        ! update min(e)
    emax(i)  = max(emax(i),exc(i))        ! update max(e)
    dimin(i) = min(dimin(i),di)           ! update min(i)
    dimax(i) = max(dimax(i),di)           ! update max(i)
!!! check for escapes.
    if (exc(i) > demax .or. semi(i) < semimin .or. semi(i) > semimax) then
      iffe = 0
      do j = 1,iesc(0)
        if (iesc(j) == i) iffe = 1
      end do
      if (iffe == 0) then
        iesc(0) = iesc(0) + 1
        iesc(iesc(0)) = i
      end if
    end if
!!! update output vector.    
    if (iout == 1) then       ! coordinates and velocities
      orele(1,i) = y0(6*(i-1)+1)*unitd
      orele(2,i) = y0(6*(i-1)+2)*unitd
      orele(3,i) = y0(6*(i-1)+3)*unitd
      orele(4,i) = y0(6*(i-1)+4)*unitd*unitt
      orele(5,i) = y0(6*(i-1)+5)*unitd*unitt
      orele(6,i) = y0(6*(i-1)+6)*unitd*unitt
    else                      ! orbital elements
      orele(1,i) = semi(i)*unitd
      orele(2,i) = exc(i)
      orele(3,i) = di
      orele(4,i) = dm
      orele(5,i) = wi
      orele(6,i) = omi
    end if
  end do
  
!!!! update minimum stepsize for integration.
  tau_min = twopi*sqrt((ascale**3)/(G*body(0)))
  stepmin = 1.0d-10*tau_min
  stepmax = 0.05*tau_min
  
!!! if requested, calculate resonant angles.
  if (jr(0) /= 0 .and. iout /= 1) then
    iah = iah + 1
    phi = cero
    iord = 0
    do i = 1,npl
      phi = phi + jr(i)*(orele(4,i) + orele(5,i) + orele(6,i))
      iord = iord + jr(i)
    end do
    phi = mod(phi,360.0d0)
    do i = 1,npl
      tita = mod(phi-iord*(orele(5,i)+orele(6,i)),360.0d0)
      if (tita.lt.cero) tita = tita + 360.0
      orele(iah,i) = tita
    end do
  end if
  
!!! search for addtional outputs (in other).
  iah = 0
  if (iom /= 0) then
    do i = 1,ntot
      other(iah+1,i) = body(i)/unitmp
    end do
    iah = iah + 1
  end if
 
!!! add output of body radius [m].
  if (ior /= 0) then
    do i = 1,ntot
      other(iah+1,i) = radius(i)*1.495978707d11
    end do
    iah = iah + 1
  end if
  
!!! add output of stellar & planetary spin.
  if (ios /= 0) then
    do i = 1,npl
      spin0 = sqrt(y(neq0+1)**2 + y(neq0+2)**2 + y(neq0+3)**2)
      spini = sqrt(y(neq0+3*i+1)**2+y(neq0+3*i+2)**2+y(neq0+3*i+3)**2)
      other(iah+1,i) = twopi/spin0
      other(iah+2,i) = twopi/spini
      other(iah+3,i) = twopi/ene(i)
    end do
    do i = npl+1,ntot
      other(iah+1,i) = cero
      other(iah+2,i) = cero
      other(iah+3,i) = twopi/ene(i)
    end do
    iah = iah + 3
  end if
  
!!! add output of orbital energy and total angular momentum.
  if (ienel /= 0) then
    do i = 1,ntot
      other(iah+1,i) = energ
      other(iah+2,i) = cang
    end do
    iah = iah + 2
  end if
  
!!! add output of chaos indicator.
  if (icout /= 0) then
    do i = 1,ntot
      ii = neq0 + neqs + 9*(i-1)
      do ij = 1,icout
        if (icaos(ij) == -3) chaos(ij,name(i)) = dimax(i) - dimin(i)
        if (icaos(ij) == -2) chaos(ij,name(i)) =  amax(i) -  amin(i)
        if (icaos(ij) == -1) chaos(ij,name(i)) =  emax(i) -  emin(i)
        if (icaos(ij) ==  1) then
          chaos(ij,name(i)) = log10(abs(y(ii+7))/(abs(time)/unitt))
          if (icalcm(i) == 1) chaos(ij,name(i)) = -1.0d0
        end if
        if (icaos(ij) == 2) then
          chaos(ij,name(i)) = y(ii+9)/abs(time)*facm
          if (icalcm(i) == 1) chaos(ij,name(i)) = chaosmax
          if (time < deltat)  chaos(ij,name(i)) = 2.0d0
          if (chaosmax > cero .and. chaos(ij,name(i)) > chaosmax) then
            icalcm(i) = 1             ! check if Megno too high
          end if
        end if
        if (icaos(ij) == -4) chaos(ij,name(i)) = tesc(name(i))
      end do
    end do
    do i = ntot+1,ntot0
      ii = neq0 + neqs + 9*(i-1)
      do ij = 1,icout
        if (icaos(ij) == -4) chaos(ij,name(i)) = tesc(name(i))
      end do
    end do
  end if
  ! 
end subroutine output


subroutine integrate (y) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),orele(7,imax),e_nofil(7,imax,-2000:2000),e_fil(7,imax)
  real*8 t_nofil(-2000:2000)
  save e_nofil,t_nofil
  ista = 1
  !
!!! if t=t0, output initial conditions.
  if (idu1 == 0) then
    time = t0
    if (ifil == 0) then
      call output (y,orele)
      tout = sgn*t0/unitt
      do i = 1,ntot
        elem(:,i) = orele(:,i)
      end do
    else
      ista = 0
      sgn = -sgn
      do i = 1,ntot
        y(6*i-2:6*i) = -y(6*i-2:6*i)
      end do
      deltm = (im+1)*deltat
      if (irtyp == 1) deltm = deltm + im*deltat
      time0 = time
      call metodo (y,deltm,orele) ! backward integration
      time = time0 - deltm
      sgn = -sgn
      do i = 1,ntot
        y(6*i-2:6*i) = -y(6*i-2:6*i)
      end do
      !
      do i = -im,im               ! forward integration from negative time
        if (time >= 0 .and. ista == 0) then
          ista = 1                ! only start chaos calculations when t >= 0
          ii = neq0 + neqs
          if (maxval(icaos(1:6)) <= 0) goto 5
          do j = 1,ntot
            y(ii+9*(j-1)+1:ii+9*(j-1)+6) = dd0
            y(ii+9*(j-1)+7:ii+9*(j-1)+9) = 1.0d-12
          end do
5         continue
        end if
        call metodo (y,deltat,orele)
        do i1 = 1,ntot
          e_nofil(:,i1,i) = orele(:,i1)
        end do
        t_nofil(i) = time
      end do
      !
      call filtrado (t_nofil,e_nofil,t_fil,e_fil)
      tout = sgn*t_fil/unitt
      do i1 = 1,ntot
        elem(:,i1) = e_fil(:,i1)
      end do
    end if
    idu1 = 1
!!! 
    if (tout <= 1.0d-10) tout = cero              ! set zero time to zero
    if (ifil == 1) tstop = tstop + (im-1)*deltat  ! increase tstop if filter
    
!!! set initial values for chaos indicators.
!    do i = 1,icout
!      ii = inout - icout + i
!      if (icaos(i) == 2)  elem(ii,1:ntot) =  dos  ! set initial Megno
!      if (icaos(i) == 1)  elem(ii,1:ntot) = -12.0 ! set initial LCE
!      if (icaos(i) == -1) elem(ii,1:ntot) =  cero ! set initial Delta_e
!      if (icaos(i) == -2) elem(ii,1:ntot) =  cero ! set initial Delta_a
!      if (icaos(i) == -3) elem(ii,1:ntot) =  cero ! set initial Delta_i
!    end do
    !
    return
  end if
  
!!! integration without filter.
  if (ifil == 0) then
    call metodo (y,deltat,orele)
    tout = sgn*time/unitt
    do i1 = 1,ntot
      elem(:,i1) = orele(:,i1)
    end do
  end if
  
!!! integration with low-pass filter.
  if (ifil == 1) then
    do i = -im,im-idecs
      do i1 = 1,ntot
        e_nofil(:,i1,i) = e_nofil(:,i1,i+idecs)
      end do
      t_nofil(i) = t_nofil(i+idecs)
    end do
    do j = 1,idecs
      call metodo (y,deltat,orele)
      i = im - idecs + j
      do i1 = 1,ntot
        e_nofil(:,i1,i) = orele(:,i1)
      end do
      t_nofil(i) = time
    end do
    call filtrado (t_nofil,e_nofil,t_fil,e_fil)
    tout = sgn*t_fil/unitt
    elem = e_fil
  end if
  !
end subroutine integrate


subroutine metodo (y,delt,orele) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),orele(7,imax)
  !
  tfinal = time + delt
  
!!! loop over total output time.
  do while (time < tfinal)
    tint = min(delt,tfinal-time)
    
!!! integrate for reduced deltat.
    if (inty == 1) call radau (y,time,tint,step)
    if (inty == 2) call bs    (y,time,tint,step)
    if (inty == 3) call rk    (y,time,tint,step)
    
!!! calculate output elements.
    call output (y,orele)
    
!!! if migration turned on, modify strength due to disk dispersal.
    if (imig_stokes == 1) then
      fac_stokes = uno2*(uno+tanh(100.0*(uno-time/t_stokes)))
    end if
    if (imig_typ1 == 1 .or. idrag == 1) then
      fac_mig = uno2*(uno+tanh(100.0*(uno-time/t_disk)))
    end if
    
!!! eliminate bodies marked as escapes by output subroutine.
    if (iesc(0) > 0) then
      call escapes (1,time,y) ; call output (y,orele)
    end if
!
  end do
    !
end subroutine metodo


subroutine force (t,y,f) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),f(18*imax),rim3(imax),ri2(imax),prodi(imax)
  common /ri3/ ri2,rim3,prodi
  !
  f(1:neq) = cero
  
!!! initialization.
  call initialize_force (y)
  
!!! calculate central mass gravitational force for each body.
  do i = 1,ntot
    Gsum = Gsum0
    if (i <= npl) Gsum = Gsum0 + G*body(i)
    i6 = 6*(i-1)
    f(i6+1:i6+3) =  y(i6+4:i6+6)
    f(i6+4:i6+6) = -Gsum*rim3(i)*y(i6+1:i6+3)
    
!!! if Lyapunov turned on, begin calculation of variational eqs.
    if (maxval(icaos(1:6)) > 0) then
      ii = neq0 + neqs + 9*(i-1)
      f(ii+1:ii+3) =  y(ii+4:ii+6)
      f(ii+4:ii+6) = -Gsum*rim3(i)*(y(ii+1:ii+3) - 3.0*y(i6+1:i6+3)*prodi(i))
      f(ii+7:ii+9) =  cero
    end if
    !    
  end do
    
!!! add perturbations.
  call perturbations (t,y,f)
  ! 
end subroutine force


subroutine initialize_force (y) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),ri2(imax),rim3(imax),prodi(imax)
  common /ri3/ ri2,rim3,prodi
  !
  do i = 1,ntot
    i6 = 6*(i-1)
    ri2(i)  = y(i6+1)*y(i6+1) + y(i6+2)*y(i6+2) + y(i6+3)*y(i6+3)
    rim2    = 1.0/ri2(i) ; rim3(i) = rim2*dsqrt(rim2)
    if (maxval(icaos(1:6)) > 0) then
      ii = neq0 + neqs + 9*(i-1)
      prodi(i) = y(i6+1)*y(ii+1) + y(i6+2)*y(ii+2) + y(i6+3)*y(ii+3)
      prodi(i) = prodi(i)/ri2(i)
    end if
    if (ri2(i) > rmax2 .or. ri2(i) < rmin2) then  ! check for escapes
      iffe = 0
      do j = 1,iesc(0)
        if (iesc(j).eq.i) iffe = 1
      end do
      if (iffe.eq.0) then
        iesc(0) = iesc(0) + 1 ; iesc(iesc(0)) = i
      end if
    end if
  end do
  !
end subroutine initialize_force


subroutine perturbations (t,y,f) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),yf(18*imax),f(18*imax),ri2(imax),rim3(imax),prodi(imax)
  real*8 kt0,kti
  common /ri3/ ri2,rim3,prodi
  !
!!! some additional perturbations are expressed in Jacobi coordinates. 
  if (imig_typ1+imig_stokes+idrag >= 1) call coord_jh (-1,yf,y)
  !
  do 20 i = 1,ntot
    Gsum = Gsum0
    if (i <= npl) Gsum = Gsum + G*body(i)
    ii = neq0 + neqs + 9*(i-1)
    i6 = 6*i     ; i5 = i6 - 1  ; i4 = i5 - 1
    i3 = i4 - 1  ; i2 = i3 - 1  ; i1 = i2 - 1
    
!!! gravitational perturbations from massive bodies (i.e. planets).
    do j = 1,npl
      if (j /= i) then
        Gsumj = G*body(j)
        j6 = 6*j     ; j5 = j6 - 1  ; j4 = j5 - 1
        j3 = j4 - 1  ; j2 = j3 - 1  ; j1 = j2 - 1
        xij = y(i1) - y(j1)
        yij = y(i2) - y(j2) 
        zij = y(i3) - y(j3)
        rij2 = xij*xij + yij*yij + zij*zij
        if (rij2 < (rhmin*(rhill(i)+rhill(j)))**2) then
          icol(0) = 1 ; icol(1) = i ; icol(2) = j
          return
        end if
        rij3 = uno/(rij2*dsqrt(rij2))
        f(i4) = f(i4) - Gsumj*(xij*rij3 + y(j1)*rim3(j))
        f(i5) = f(i5) - Gsumj*(yij*rij3 + y(j2)*rim3(j))
        f(i6) = f(i6) - Gsumj*(zij*rij3 + y(j3)*rim3(j))
        
!!! finish calculation of variational eqs for LCE.
        if (maxval(icaos(1:6)) > 0) then
          if (i <= npl) then
            jj = neq0 + neqs + 9*(j-1)
            prodij = xij*(y(ii+1)-y(jj+1)) + yij*(y(ii+2)-y(jj+2))
            prodij = prodij + zij*(y(ii+3)-y(jj+3))
            prodij = prodij/rij2
            f(ii+4)= f(ii+4) - Gsumj*rij3*(y(ii+1)-y(jj+1)-3.*xij*prodij)
            f(ii+5)= f(ii+5) - Gsumj*rij3*(y(ii+2)-y(jj+2)-3.*yij*prodij)
            f(ii+6)= f(ii+6) - Gsumj*rij3*(y(ii+3)-y(jj+3)-3.*zij*prodij)
            f(ii+4)= f(ii+4) - Gsumj*rim3(j)*(y(jj+1)-3.0*y(j1)*prodi(j))
            f(ii+5)= f(ii+5) - Gsumj*rim3(j)*(y(jj+2)-3.0*y(j2)*prodi(j))
            f(ii+6)= f(ii+6) - Gsumj*rim3(j)*(y(jj+3)-3.0*y(j3)*prodi(j))
          else
            jj = neq0 + neqs + 9*(j-1)
            prodij = (xij*y(ii+1) + yij*y(ii+2) + zij*y(ii+3))/rij2
            f(ii+4) = f(ii+4) - Gsumj*rij3*(y(ii+1)-3.0*xij*prodij)
            f(ii+5) = f(ii+5) - Gsumj*rij3*(y(ii+2)-3.0*yij*prodij)
            f(ii+6) = f(ii+6) - Gsumj*rij3*(y(ii+3)-3.0*zij*prodij)
          end if
        end if
        !  
      end if
    end do
    
!!! add J2 oblateness term.
    if (dj2mod > cero) then
      term = 1.5*Gsum*dj2mod/ri2(i)/ri2(i)/sqrt(ri2(i))
      z2r2 = y(i3)*y(i3)/ri2(i)
      f(i4) = f(i4) - term*y(i1)*(uno - 5.0*z2r2)
      f(i5) = f(i5) - term*y(i2)*(uno - 5.0*z2r2)
      f(i6) = f(i6) - term*y(i3)*(3.0 - 5.0*z2r2)
    end if
    
!!! check if additional perturbations are called for.
    if (irel+itid+iyar >= 1) then
      vi2  = y(i4)*y(i4) + y(i5)*y(i5) + y(i6)*y(i6)
      rv   = y(i1)*y(i4) + y(i2)*y(i5) + y(i3)*y(i6)
      ri   = dsqrt(ri2(i))
      rvri = rv/ri2(i)
    end if
    if (imig_typ1+imig_stokes+idrag >= 1) then
      rvf  = yf(i1)*yf(i4) + yf(i2)*yf(i5) + yf(i3)*yf(i6)
      rfi2 = yf(i1)*yf(i1) + yf(i2)*yf(i2) + yf(i3)*yf(i3)
      rfi  = sqrt(rfi2)
    end if
    
!!! add relativistic effects (Richardson & Kelly, 1988 CeMDA 43, 193-210).
    if (i <= npl .and. irel == 1) then
      sig  = body(i)*body(0)/(body(i)+body(0))**2
      sig0 = (uno-3.0*sig)/8.0d0
      sig1 =  uno2*Gsum*(3.0+sig)
      sig2 = -uno2*Gsum*Gsum
      sig3 =  uno2*Gsum*sig
      cte1 =  4.0*sig0*vi2 + dos*sig1/ri 
      cte2 =  dos*sig3*rv*rim3(i)
      fx_rel = cte1*y(i4) + cte2*y(i1)
      fy_rel = cte1*y(i5) + cte2*y(i2)
      fz_rel = cte1*y(i6) + cte2*y(i3)
      px = y(i4) + fx_rel*unoc2
      py = y(i5) + fy_rel*unoc2
      pz = y(i6) + fz_rel*unoc2
      ctev1 =  dos*sig3*rv*rim3(i)
      ctev2 = -sig1*vi2*rim3(i) - dos*sig2*rim3(i)/ri
      ctev2 =  ctev2 - 3.0*sig3*rv*rv*rim3(i)/ri2(i)
      fvx_rel = (ctev1*px + ctev2*y(i1))*unoc2
      fvy_rel = (ctev1*py + ctev2*y(i2))*unoc2
      fvz_rel = (ctev1*pz + ctev2*y(i3))*unoc2
      f(i1) = f(i1) - fx_rel*unoc2
      f(i2) = f(i2) - fy_rel*unoc2
      f(i3) = f(i3) - fz_rel*unoc2
      f(i4) = f(i4) + fvx_rel
      f(i5) = f(i5) + fvy_rel
      f(i6) = f(i6) + fvz_rel
    end if
    
!!! add Stokes-type planetary migration.
    if (i <= npl .and. imig_stokes == 1 .and. t <= 1.2*t_stokes) then
      if (coefc(i) /= cero) then
        vcirc  =  dsqrt(Gsum/rfi)
        vxcirc = -vcirc*yf(i2)/rfi
        vycirc =  vcirc*yf(i1)/rfi
        f(i4) = f(i4) - fac_stokes*coefc(i)*(yf(i4) - alfa(i)*vxcirc)
        f(i5) = f(i5) - fac_stokes*coefc(i)*(yf(i5) - alfa(i)*vycirc)
        f(i6) = f(i6) - fac_stokes*coefc(i)*yf(i6)
      end if
    end if
    
!!! add Type-I planetary migration (Goldreich & Schlichting 2014).
    if (i <= npl .and. imig_typ1 == 1 .and. t <= 1.2*t_disk) then
      Hr  = Hr0*(rfi**flare)/sqrt(body(0))
      funcg = uno
      if (icav /= 0) then
        piso  = 1.0d-3
        cteg  = uno2*log(piso)
        tanhx = tanh((rfi-ric)/delta_ic)
        logfg = min(dos,(uno+piso)*((uno-tanhx)-uno2)+uno2)
        funcg = min(uno,exp(cteg*logfg)-piso)
      end if
      sigma = sigma0*funcg/(rfi**denpot) ! surface density at r=ri
      twave = (body(0)**2)*(Hr**4)/body(i)/sigma/semi(i)/semi(i)/ene(i)
      Q_a   = uno/(2.7 + 1.1*denpot)
      tau_a0 = twave*Q_a/(Hr**2)
      tau_e  = twave*Q_e/0.78d0
      tau_a  = uno/(uno/tau_a0 + 2.0*ang_fac*exc(i)*exc(i)/tau_e)
      tau_i  = twave/0.5440
      rvri  = rvf/rfi2
      f(i4) = f(i4) - fac_mig*(yf(i4)/dos/tau_a + dos*rvri*yf(i1)/tau_e)
      f(i5) = f(i5) - fac_mig*(yf(i5)/dos/tau_a + dos*rvri*yf(i2)/tau_e)
      f(i6) = f(i6) - fac_mig*(yf(i6)/tau_i)
    end if
    
!!! tides raised on star & planet (Darwin-Mignard model).
    if (i <= npl .and. itid == 1) then
      zmred = body(i)*body(0)/(body(i) + body(0))
!!! tidal precession term.
      term = love(0)*(body(i)/body(0))*(radius(0)**5)
      term = term + love(i)*(body(0)/body(i))*(radius(i)**5)
      term = 3.0*Gsum*term/(ri2(i)**4)
      f(i4) = f(i4) - term*y(i1)
      f(i5) = f(i5) - term*y(i2)
      f(i6) = f(i6) - term*y(i3)
!!! stellar tide (dissipative term).
      cte0 = -3.0*k2delt0(i)*(body(i)/body(0))*((radius(0)/ri2(i))**5)
      romx = y(i2)*y(neq0+3) - y(i3)*y(neq0+2)
      romy = y(i3)*y(neq0+1) - y(i1)*y(neq0+3)
      romz = y(i1)*y(neq0+2) - y(i2)*y(neq0+1)
      fvx_tid = cte0*(dos*y(i1)*rv + ri2(i)*(romx + y(i4)))
      fvy_tid = cte0*(dos*y(i2)*rv + ri2(i)*(romy + y(i5)))
      fvz_tid = cte0*(dos*y(i3)*rv + ri2(i)*(romz + y(i6)))
      f(i4) = f(i4) + fac_tid*fvx_tid
      f(i5) = f(i5) + fac_tid*fvy_tid
      f(i6) = f(i6) + fac_tid*fvz_tid
      tx = y(i2)*fvz_tid - y(i3)*fvy_tid  
      ty = y(i3)*fvx_tid - y(i1)*fvz_tid
      tz = y(i1)*fvy_tid - y(i2)*fvx_tid
      f(neq0+1) = f(neq0+1) - fac_tid*tx*zmred/zmi(0)! ODEs for stellar rotation
      f(neq0+2) = f(neq0+2) - fac_tid*ty*zmred/zmi(0)
      f(neq0+3) = f(neq0+3) - fac_tid*tz*zmred/zmi(0)
!!! planetary tide (dissipative term).
      ctep = -3.0*k2delti(i)*(body(0)/body(i))*((radius(i)/ri2(i))**5)
      romx = y(i2)*y(neq0+3*i+3) - y(i3)*y(neq0+3*i+2)
      romy = y(i3)*y(neq0+3*i+1) - y(i1)*y(neq0+3*i+3)
      romz = y(i1)*y(neq0+3*i+2) - y(i2)*y(neq0+3*i+1)
      fvx_tid = ctep*(dos*y(i1)*rv + ri2(i)*(romx + y(i4)))
      fvy_tid = ctep*(dos*y(i2)*rv + ri2(i)*(romy + y(i5)))
      fvz_tid = ctep*(dos*y(i3)*rv + ri2(i)*(romz + y(i6)))
      f(i4) = f(i4) + fac_tid*fvx_tid
      f(i5) = f(i5) + fac_tid*fvy_tid
      f(i6) = f(i6) + fac_tid*fvz_tid
      tx = y(i2)*fvz_tid - y(i3)*fvy_tid
      ty = y(i3)*fvx_tid - y(i1)*fvz_tid
      tz = y(i1)*fvy_tid - y(i2)*fvx_tid
      f(neq0+3*i+1) = -fac_tid*tx*zmred/zmi(i) ! ODEs for planet rotation
      f(neq0+3*i+2) = -fac_tid*ty*zmred/zmi(i)
      f(neq0+3*i+3) = -fac_tid*tz*zmred/zmi(i)
    end if
    
!!! non-linear aerodynamic drag on massless particles.
    if (i > npl .and. idrag == 1 .and. t <= 2.0*t_disk) then
      funcg = uno
      cteg  = uno
      if (icav /= 0) then
        cteg  = uno2*log(1.0d-3)
        tanhx = tanh((rfi-ric)/delta_ic)
        funcg = exp(cteg*(uno - tanhx))
      end if
      sigma = sigma0*funcg/(rfi**denpot)  ! surface density at r=ri
      lstar = uno                         ! stellar luminosity [Lsol]
      Hr    = Hr0*(rfi**flare)*(lstar**(1.0d0/8.0d0))/sqrt(body(0))
      rho_0 = unor2pi*sigma/Hr/rfi
      rho_z = rho_0
      if (abs(y(i3)) > 1.0d-3*rfi) then
        rho_z = rho_0*exp(-uno2*y(i3)*y(i3)/(Hr*Hr*rfi*rfi))
      end if
      coef = coefc(i)*rho_z
      wgas = mod(ggas*t + wgas0,twopi)
      anomf = datan2(yf(i2),yf(i1)) - wgas
      pg = rfi ; if (egas > 1.0d-3) pg = rfi*(1.0d0+egas*dcos(anomf))
      vr0 =  alfa(i)*dsqrt(Gsum0/pg)
      vx  = -sgn*vr0*(sin(anomf+wgas) + egas*sin(wgas))
      vy  =  sgn*vr0*(cos(anomf+wgas) + egas*cos(wgas))
      vrel = sqrt((yf(i4)-vx)**2 + (yf(i5)-vy)**2 + yf(i6)**2)
      f(i4) = f(i4) - fac_mig*coef*vrel*(yf(i4) - vx)
      f(i5) = f(i5) - fac_mig*coef*vrel*(yf(i5) - vy)
      f(i6) = f(i6) - fac_mig*coef*vrel*yf(i6)
    end if
    
!!! add (modeled) Yarkovsky to massless particles.
    if (i > npl .and. iyar == 1) then
      unoa = 2.0d0/ri - vi2/Gsum0
      facy = dadty(i)*0.5*Gsum0*unoa*unoa/vi2
      f(i4) = f(i4) + facy*y(i4)
      f(i5) = f(i5) + facy*y(i5)
      f(i6) = f(i6) + facy*y(i6)
    end if
    
!!! add relativistic effects to massless particles.
    if (i > npl .and. irelp == 1) then
      cte1 =  (uno2*vi2 + 3.0*Gsum/ri)*unoc2
      ctev = -(1.5*Gsum*vi2*rim3(i) + Gsum*Gsum*rim3(i)/ri)*unoc2
      f(i1) = f(i1) - cte1*y(i4)
      f(i2) = f(i2) - cte1*y(i5)
      f(i3) = f(i3) - cte1*y(i6)
      f(i4) = f(i4) + cte2*y(i1)
      f(i5) = f(i5) + cte2*y(i2)
      f(i6) = f(i6) + cte2*y(i3)
    end if

20 end do
  
!!! calculate Lyapunov & Megno.
  if (maxval(icaos(1:6)).gt.0) then
    prd = cero
    dis = cero
    do i = 1,npl
      if (icalcm(i).eq.0) then
        ii = neq0 + neqs + 9*(i-1)
        prd = prd + y(ii+1)*f(ii+1) + y(ii+2)*f(ii+2) + y(ii+3)*f(ii+3)
        prd = prd + y(ii+4)*f(ii+4) + y(ii+5)*f(ii+5) + y(ii+6)*f(ii+6)
        dis = dis + y(ii+1)*y(ii+1) + y(ii+2)*y(ii+2) + y(ii+3)*y(ii+3)
        dis = dis + y(ii+4)*y(ii+4) + y(ii+5)*y(ii+5) + y(ii+6)*y(ii+6)
      end if
    end do
    do i = 1,npl
      if (icalcm(i).eq.0) then
        ii = neq0 + neqs + 9*(i-1)
        f(ii+7) = ista*prd/dis           ! LCE
        f(ii+8) = f(ii+7)*(t/unitt)/facm ! Megno
        if (t.gt.uno) f(ii+9) = dos*y(ii+8)/(t/unitt) ! averaged Megno
      end if
    end do
    do i = npl+1,ntot
      if (icalcm(i).eq.0) then
        ii = neq0 + neqs + 9*(i-1)
        prd = y(ii+1)*f(ii+1) + y(ii+2)*f(ii+2) + y(ii+3)*f(ii+3)
        prd = prd + y(ii+4)*f(ii+4) + y(ii+5)*f(ii+5) + y(ii+6)*f(ii+6)
        dis = y(ii+1)*y(ii+1) + y(ii+2)*y(ii+2) + y(ii+3)*y(ii+3)
        dis = dis + y(ii+4)*y(ii+4) + y(ii+5)*y(ii+5) + y(ii+6)*y(ii+6)
        f(ii+7) = ista*prd/dis           ! LCE
        f(ii+8) = f(ii+7)*(t/unitt)/facm ! Megno
        if (t.gt.uno) f(ii+9) = dos*y(ii+8)/(t/unitt)  ! averaged Megno
      end if
    end do
  end if
  !
end subroutine perturbations


subroutine filtrado (t_nofil,e_nofil,t_fil,e_fil) ; use common
  implicit real*8 (a-h,k-z)
  real*8 e_nofil(7,imax,-2000:2000),e_fil(7,imax),e_fil_c(7,imax)
  real*8 t_nofil(-2000:2000),e_fil_s(7,imax)
  !
  do i1 = 1,ntot
    do i2 = 1,inout0
      if (iang(i2) == 1) then ! it is an angle
        e_fil_c(i2,i1) = cero ; e_fil_s(i2,i1) = cero
        do j = -im,im
          termc = cos(e_nofil(i2,i1,j)*rad)
          terms = sin(e_nofil(i2,i1,j)*rad)
          e_fil_c(i2,i1) = e_fil_c(i2,i1) + termc*airy(abs(j))
          e_fil_s(i2,i1) = e_fil_s(i2,i1) + terms*airy(abs(j))
        end do
        e_fil(i2,i1) = datan2(e_fil_s(i2,i1),e_fil_c(i2,i1))/rad
        if (e_fil(i2,i1) < 1.0d-5) e_fil(i2,i1) = e_fil(i2,i1) + 360.0
      else                    ! it is not an angle
        e_fil(i2,i1) = cero
        do j = -im,im
          e_fil(i2,i1) = e_fil(i2,i1) + e_nofil(i2,i1,j)*airy(abs(j))
        end do
      end if
    end do
    do i2 = inout0+1,inout
      e_fil(i2,i1) = e_nofil(i2,i1,0)
    end do
  end do
  !
  t_fil = t_nofil(0)
  !
end subroutine filtrado


subroutine filter ; use common
  implicit real*8 (a-h,k-z)
  real*8 e(-2000:2000),cuad(0:2000)
  !
  open (45,file='filter.dat',status='replace')
  !
  dnyq  = uno2/deltat         ! critical (nyquist) frequency
  wpass = dnyq/dfloat(idec)   ! pass frequency
  
!!! time interval associated to filter.
  tmax = 0.1*dfloat(im-1)/wpass
  
!!! step function.
  e = cero
  do iw = -im,im
    w = dfloat(iw)/tmax ; e(iw) = uno
  end do
  
!!! filter = anti-transform of step function.
  area = cero
  do i = 0,im
    airy(i) = cero
    ti = dfloat(i)*deltat
    do iw = -im,im
      w = dfloat(iw)/tmax ; airy(i) = airy(i) + e(iw)*dcos(w*ti)
    end do
    area = area + airy(i)
    if (i > 0) area = area + airy(i)
  end do
  airy = airy/area
  
!!! rehydrated step function.
  do i = 0,2*im
    cuad(i) = cero
    ti = dfloat(i)*deltat
    do iw = 0,im
      w = dfloat(iw)/tmax
      cuad(i) = cuad(i) + airy(iw)*dcos(w*ti)
      if (iw /= 0) cuad(i) = cuad(i) + airy(iw)*dcos(w*ti)
    end do
    write (45,*) i,airy(i)/airy(0),cuad(i)
  end do
  !
  close (45)
  !
end subroutine filter


subroutine collision (time_col,y) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),mrat_ic,mrat_jc
  !
  if (ienc_first == 0 .and. iencout == 1) then
    open (16,file=arch_enc,status='replace')
    ienc_first = 1
  end if
  
!!! identify bodies that collided.
  ic = min(icol(1),icol(2))
  jc = max(icol(1),icol(2))
  bodyt = body(ic) + body(jc)
  
!!! record collision in encounter file.
  if (time_col >= cero .and. iencout == 1) then 
    write (16,77) name(ic),name(jc),time_col/365.2563
77  format ('collision between bodies ',2i5,'   at T =',1pd15.7,'.')
    flush (16)
  end if
  !
  mrat_ic = body(ic)/bodyt
  mrat_jc = body(jc)/bodyt
  
!!! accrete bodies using conservation of linear momentum, and place in i=ic.
  y(6*ic-5) = y(6*ic-5)*mrat_ic + y(6*jc-5)*mrat_jc
  y(6*ic-4) = y(6*ic-4)*mrat_ic + y(6*jc-4)*mrat_jc
  y(6*ic-3) = y(6*ic-3)*mrat_ic + y(6*jc-3)*mrat_jc
  y(6*ic-2) = y(6*ic-2)*mrat_ic + y(6*jc-2)*mrat_jc
  y(6*ic-1) = y(6*ic-1)*mrat_ic + y(6*jc-1)*mrat_jc
  y(6*ic-0) = y(6*ic-0)*mrat_ic + y(6*jc-0)*mrat_jc
  
!!! same with components of planetary spin.
  y(neq0+3*ic+1) = y(neq0+3*ic+1)*mrat_ic + y(neq0+3*jc+1)*mrat_jc
  y(neq0+3*ic+2) = y(neq0+3*ic+2)*mrat_ic + y(neq0+3*jc+2)*mrat_jc
  y(neq0+3*ic+3) = y(neq0+3*ic+3)*mrat_ic + y(neq0+3*jc+3)*mrat_jc
  
!!! update radius and (in case of particles) drag coefficient.
  if (min(radius(ic),radius(jc)) > 1.0d-20) then
    rhoi = body(ic)/(radius(ic)**3)
    rhoj = body(jc)/(radius(jc)**3)
    rhom = (rhoi*body(ic) + rhoj*body(jc))/bodyt 
    radius(ic) = (bodyt/rhom)**0.3333333333
    rho(ic)    = (rho(ic)*body(ic) + rho(jc)*body(jc))/bodyt 
    if (ic > npl) coefc(ic) = (3.0/8.0)*0.44/radius(ic)/rhom
  end if
  
!!! update mass and Yarkovsky effect.
  body(ic) = bodyt
!!!  dadty(ic) = 
  
!!! save collision times.
  tesc(ic) = time_col/365.2563d0
  tesc(jc) = time_col/365.2563d0
  
!!! eliminate secondary body and update all remaining variables.
  iesc(0) = 1
  iesc(1) = jc
  false_time = -10.0
  call escapes (-1,false_time,y)

!!! set chaos indicators to values associated with orbital instability.
  do i = 1,icout
    if (icaos(i) ==  2) chaos(i,name(ic)) = chaosmax
    if (icaos(i) ==  2) chaos(i,name(jc)) = chaosmax
    if (icaos(i) ==  1) chaos(i,name(ic)) = -1.0
    if (icaos(i) ==  1) chaos(i,name(jc)) = -1.0
    if (icaos(i) == -1) chaos(i,name(ic)) = 1.0d2
    if (icaos(i) == -1) chaos(i,name(jc)) = 1.0d2
    if (icaos(i) == -2) chaos(i,name(ic)) = 1.0d2
    if (icaos(i) == -2) chaos(i,name(jc)) = 1.0d2
    if (icaos(i) == -3) chaos(i,name(ic)) = 1.0d2
    if (icaos(i) == -3) chaos(i,name(jc)) = 1.0d2
  end do
  
!!! reset collision flag to zero.
  icol(0) = 0
  !
end subroutine collision


subroutine escapes (inocol,time_esc,y) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),xc(6),msum,dist(imax)
  logical mask(imax)
  !
  if (ienc_first == 0 .and. iencout == 1) then
    open (16,file=arch_enc,status='replace')
    ienc_first = 1
  end if
  !
  do 10 j = 1,iesc(0)
    iesj = iesc(j)
    
!!! save escape time.
    if (time_esc >= cero) tesc(iesj) = time_esc/365.2563

!!! if Shannon entropy rquested and time < tmin_shan, set tesc_S = tesc.
    if (ishannon > 0) then
!    if (ishannon > 0 .and. time < tmin_shan) then
      tesc_S(1:npl0) = time_esc/365.2563
    end if
    
!!! record escape in encounter file.
    if (time_esc >= cero .and. iencout == 1) then
      msum = body(0) + body(iesj)
      xc(1:6) = y(6*iesj-5:6*iesj)
      call elements (msum,xc,a,e,dinc,capm,omega,capom)
      write (16,77) name(iesj),time_esc/365.2563,a,e
77    format ('escape of body ',i5,',   at T =',1pd15.7,'.   a = ' &
           ,d11.4,'   e = ',d11.4)
      flush (16)
    end if
    
!!! set lyapunov & megno to maximum values.
    if (icout > 0 .and. inocol > 0) then
      do i = 1,icout
        if (icaos(i) ==  2) chaos(i,name(iesj)) = chaosmax
        if (icaos(i) ==  1) chaos(i,name(iesj)) = -1.0
        if (icaos(i) == -1) chaos(i,name(iesj)) = 1.0d2
        if (icaos(i) == -2) chaos(i,name(iesj)) = 1.0d2
        if (icaos(i) == -3) chaos(i,name(iesj)) = 1.0d2
      end do
    end if
    
!!! eliminate body from list.
    do i = iesj,ntot-1
      i1 = i + 1
      do jj = 5,0,-1
        y(6*i-jj) = y(6*i1-jj)
      end do
      if (maxval(icaos(1:6)) > 0) then
        ii  = neq0 + neqs + 9*(i-1)
        ii1 = neq0 + neqs + 9*(i1-1)
        y(ii+1:ii+9) = y(ii1+1:ii1+9)
      end if
      eleini(1:7,i) = eleini(1:7,i1)
      body(i)   = body(i1)   ; name(i)  = name(i1)
      radius(i) = radius(i1) ; rho(i)   = rho(i1)
      coefc(i)  = coefc(i1)  ; alfa(i)  = alfa(i1)
      prot(i)   = prot(i1)   ; dadty(i) = dadty(i1)
      amin(i)   = amin(i1)   ; amax(i)  = amax(i1)
      emin(i)   = emin(i1)   ; emax(i)  = emax(i1)
      dimin(i)  = dimin(i1)  ; dimax(i) = dimax(i1)
      icalcm(i) = icalcm(i1)
      if (iesj <= npl) then
        love(i) = love(i1) ; k2delt0(i) = k2delt0(i1) ; k2delti(i) = k2delti(i1)
        zmi(i)  = zmi(i1)  ; fac_zmi(i) = fac_zmi(i1)
      end if
    end do
    
!!! update number of bodies & equations.
    if (iesj <= npl) then
      npl = npl - 1
    else
      npart = npart - 1
    end if
    ntot = ntot - 1
    neq0 = 6*ntot
    neq  = neq0 + neqs + neqlyp  ! partially updated total number of equations
    
!!! shift remaining equations.
    do i = neq0+1,neq ; y(i) = y(i+6) ; end do

!!! if Jacobi reference system used, update body order and mass factors.
    if (ietyp == 3) then
      dist = 1.0d30
      dist(1:ntot) = eleini(1,1:ntot)
      mask(1:ntot) = .true.
      iorder(0) = 0
      do i = 1,ntot
        iorder(i) = minloc(dist,1,mask)
        mask(iorder(i)) = .false.
      end do
      eta(0) = body(0)
      do i = 1,ntot
        eta(iorder(i)) = eta(iorder(i-1)) + body(iorder(i))
      end do
    end if
    
!!! eliminate spin equations if escapee was planet & shift equations.
    if (neqs > 0 .and. iesj < npl) then
      do i = neq0+3+3*(iesj-1)+1,neq ; y(i) = y(i+3) ; end do
    end if
    
!!! final update on number of equations.
    if (itid == 1) neqs = 3*(npl+1)              ! number of spin equations
    if (maxval(icaos(1:6)) > 0) neqlyp = 9*ntot  ! number of var. equations
    neq = neq0 + neqs + neqlyp                   ! total number of equations
    
!!! update remaining escapee list.
    do jj = j+1,iesc(0)
      if (iesc(jj) > iesj) iesc(jj) = iesc(jj) - 1
    end do
    !
10 end do
11 continue
  iesc(0) = 0
  !
end subroutine escapes


subroutine coord (msum,a,e,inc,capm,omega,capom,xc) ; use common
  implicit real*8 (a-h,k-z)
  real*8 inc,xc(6)
  !  
  gm  = G*msum
  em1 = e - uno
  
!!! generate rotation matrices (on p. 42 of fitzpatrick)
  sp = sin(omega*rad) ; cp = cos(omega*rad)
  so = sin(capom*rad) ; co = cos(capom*rad)
  si = sin(inc*rad)   ; ci = cos(inc*rad)
  d11 =  cp*co - sp*so*ci ; d12 =  cp*so + sp*co*ci ; d13 = sp*si
  d21 = -sp*co - cp*so*ci ; d22 = -sp*so + cp*co*ci ; d23 = cp*si
  
!!! calculate coordinates and velocities.
  call aver (capm,e,cape,dummy)
  scap  = sin(cape*rad)  ; ccap  = cos(cape*rad)
  sqe   = sqrt(uno - e*e)
  sqgma = sqrt(gm*a)
  xfac1 = a*(ccap - e)   ; xfac2 = a*sqe*scap
  ri    = uno/(a*(uno - e*ccap))
  vfac1 = -ri*sqgma*scap ; vfac2 = ri*sqgma*sqe*ccap
  !  
  xc(1) = d11*xfac1 + d21*xfac2 ; xc(2) = d12*xfac1 + d22*xfac2
  xc(3) = d13*xfac1 + d23*xfac2 ; xc(4) = d11*vfac1 + d21*vfac2
  xc(5) = d12*vfac1 + d22*vfac2 ; xc(6) = d13*vfac1 + d23*vfac2
  !
end subroutine coord


subroutine elements (msum,xc,a,e,inc,capm,omega,capom) ; use common
  implicit real*8 (a-h,k-z)
  real*8 xc(6),inc
  !
  Gmsum = G*msum
  x  = xc(1) ; y  = xc(2) ; z  = xc(3)
  vx = xc(4) ; vy = xc(5) ; vz = xc(6)
  
!!! compute the angular momentum h, and thereby the inclination inc.
  hx = y*vz - z*vy ; hy = z*vx - x*vz
  hz = x*vy - y*vx ; h2 = hx*hx + hy*hy + hz*hz
  h  = sqrt(h2)    ; inc = acos(hz/h)
  
!!! compute longitude of ascending node capom and the argument of latitude u.
  fac = sqrt(hx**2 + hy**2)/h
  if (fac < error) then
    capom = cero ; u = atan2(y,x)
    if (abs(inc-pi) < 10.d0*error) u = -u
  else
    capom = atan2(hx,-hy) ; u = atan2(z/sin(inc),x*cos(capom)+y*sin(capom))
  end if
  if (capom < cero) capom = capom + twopi ; if (u < cero) u = u + twopi
  
!!! compute the radius r, vel. squared v2, the dot product rdotv & energy.
  r  = sqrt(x*x + y*y + z*z)
  v2 = vx*vx + vy*vy + vz*vz ; v = sqrt(v2)
  vdotr  = x*vx + y*vy + z*vz
  energy = uno2*v2 - Gmsum/r
  
!!! determine type of conic section and label it via ialpha
  if (abs(energy*r/gmsum) < sqrt(error)) then
    ialpha = 0
  else
    if (energy < cero) ialpha = -1 
    if (energy > cero) ialpha = +1
  endif
  
!!! depending on the conic type, determine the remaining elements
!!! ellipse :
  if (ialpha == -1) then
    a = -uno2*Gmsum/energy  
    fac = uno - h2/(gmsum*a)
    if (fac> error) then
      e = sqrt(fac)
      face = (a-r)/(a*e)
      if (face > uno) then
        cape = cero
      else
        if (face > -1.d0) then
          cape = acos(face)
        else
          cape = pi
        end if
      end if
      if (vdotr < cero) cape = twopi - cape
      cw = (cos(cape)-e)/(uno - e*cos(cape))
      sw = sqrt(uno-e*e)*sin(cape)/(uno-e*cos(cape))
      w  = atan2(sw,cw)
      if (w < cero) w = w + twopi
    else
      e = cero ; w = u ; cape = u
    end if
    capm = cape - e*sin(cape)
    omega = u - w
    if (omega < cero) omega = omega + twopi
    omega = omega - int(omega/twopi)*twopi
  end if
  
!!! hyperbola
  if (ialpha == +1) then
    a = +uno2*Gmsum/energy  
    fac = h2/(Gmsum*a)
    if (fac > error) then
      e = sqrt(uno+fac)
      tmpf = (a+r)/(a*e)
      if (tmpf < uno) tmpf = uno
      capf = log(tmpf+sqrt(tmpf*tmpf-uno))
      if (vdotr < cero) capf = -capf
      cw = (e-cosh(capf))/(e*cosh(capf)-uno)
      sw = sqrt(e*e-uno)*sinh(capf)/(e*cosh(capf)-uno)
      w = atan2(sw,cw)
      if (w < cero) w = w + twopi
    else
      e = uno
      tmpf = uno2*h2/gmsum
      w = acos(dos*tmpf/r-uno)
      if (vdotr < cero) w = twopi - w
      tmpf = (a+r)/(a*e)
      capf = log(tmpf+sqrt(tmpf*tmpf-uno))
    end if
    capm = e*sinh(capf) - capf
    omega = u - w
    if (omega < cero) omega = omega + twopi
    omega = omega - int(omega/twopi)*twopi
  end if
  
!!! parabola : ( note - in this case we use "a" to mean pericentric distance)
  if (ialpha == 0) then
    a = uno2*h2/Gmsum  ; e = uno ; w = acos(dos*a/r-uno)
    if (vdotr < cero) w = twopi - w
    tmpf = tan(uno2*w)
    capm = tmpf*(uno+tmpf*tmpf/3.d0)
    omega = u - w
    if (omega < cero) omega = omega + twopi
    omega = omega - int(omega/twopi)*twopi
  end if

!!! change angles to degrees.
  inc = inc/rad ; capm = capm/rad ; omega = omega/rad ; capom = capom/rad  
  ! 
end subroutine elements

      
subroutine energia (y,energ,c) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),y0(18*imax)
  common /cvs/ xstar,ystar,zstar,vxstar,vystar,vzstar
  ! 
  call coord_bh (-1,y0,y)
  !
  in6 = 6*ntot
  cx = (y0(in6-4)*y0(in6-0) - y0(in6-3)*y0(in6-1))*body(ntot)
  cy = (y0(in6-3)*y0(in6-2) - y0(in6-5)*y0(in6-0))*body(ntot)
  cz = (y0(in6-5)*y0(in6-1) - y0(in6-4)*y0(in6-2))*body(ntot)
  !
  enerk = uno2*body(ntot)*(y0(in6-2)**2 + y0(in6-1)**2 + y0(in6-0)**2)
  enerp = cero
  do i = 1,ntot-1
    i6 = 6*i     ;  i5 = i6 - 1
    i4 = i5 - 1  ;  i3 = i4 - 1
    i2 = i3 - 1  ;  i1 = i2 - 1
    cx = cx + (y0(i2)*y0(i6) - y0(i3)*y0(i5))*body(i)
    cy = cy + (y0(i3)*y0(i4) - y0(i1)*y0(i6))*body(i)
    cz = cz + (y0(i1)*y0(i5) - y0(i2)*y0(i4))*body(i)
    enerk = enerk + uno2*body(i)*(y0(i4)**2 + y0(i5)**2 + y0(i6)**2)
    do j = i+1,ntot
      rr2 = (y0(6*i-5)-y0(6*j-5))**2 + (y0(6*i-4)-y0(6*j-4))**2
      rr2 = rr2 + (y0(6*i-3)-y0(6*j-3))**2 
      enerp = enerp - body(i)*body(j)/dsqrt(rr2)
    end do
  end do
      
!!! add energy and angular momentum of star.
  cx = cx + (ystar*vzstar-zstar*vystar)*body(0)
  cy = cy + (zstar*vxstar-xstar*vzstar)*body(0)
  cz = cz + (xstar*vystar-ystar*vxstar)*body(0)
  enerk = enerk + uno2*body(0)*(vxstar**2 + vystar**2 + vzstar**2)
  do j = 1,ntot
    rr2 = (xstar-y0(6*j-5))**2 + (ystar-y0(6*j-4))**2
    rr2 = rr2 + (zstar-y0(6*j-3))**2 
    enerp = enerp - body(0)*body(j)/dsqrt(rr2)
  end do
  !
  energ = enerk + G*enerp  ! = dsqrt(cx*cx + cy*cy + cz*cz)
  
!!! add spin energy & angular momentum (not implemented).
  energ_s = cero
  c_s     = cero
  energ   = energ + energ_s 
  !
end subroutine energia


subroutine coord_jh (idir,yj,ya) ; use common
  implicit real*8 (a-h,k-z)
  real*8 yj(18*imax),ya(18*imax),ytemp(18*imax),y0(6)
  !
  if (idir.eq.1) then   ! convert jacobi to astrocentric
    ya(1:6*ntot) = cero
    ior1 = 6*(iorder(1)-1)
    ya(ior1+1:ior+6) = yj(ior1+1:ior1+6)
    do i = 2,ntot
      ior1 = 6*(iorder(i)-1)
      ior0 = 6*(iorder(i-1)-1)
      ya(ior1+1:ior1+6) = yj(ior1+1:ior1+6) + &
           (eta(iorder(i-2))*(ya(ior0+1:ior0+6) - yj(ior0+1:ior0+6)) + &
           body(iorder(i-1))*ya(ior0+1:ior0+6))/eta(iorder(i-1))
    end do
  end if
  !
  if (idir.eq.-1) then      ! convert astrocentric to jacobi
    yj(1:6*ntot) = cero
    ior1 = 6*(iorder(1)-1)
    yj(ior1+1:ior1+6) = ya(ior1+1:ior1+6)
    do i = 2,ntot
      ior1 = 6*(iorder(i)-1)
      ior0 = 6*(iorder(i-1)-1)
      yj(ior1+1:ior1+6) = ya(ior1+1:ior1+6) - &
           (eta(iorder(i-2))*(ya(ior0+1:ior0+6) - yj(ior0+1:ior0+6)) + &
           body(iorder(i-1))*ya(ior0+1:ior0+6))/eta(iorder(i-1))
    end do
  end if
  !
end subroutine coord_jh


subroutine coord_bh (idir,y0,y) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y0(18*imax),y(18*imax)
  common /cvs/ xbase,ybase,zbase,vxbase,vybase,vzbase
  !
  xbase  = cero  ;  ybase  = cero  ;  zbase  = cero
  vxbase = cero  ;  vybase = cero  ;  vzbase = cero
  ! 
  if (idir == 1) then   ! convert barycentric to astrocentric
    
!!! determine barycentric position & velocity of star.
    do i = 1,npl
      i61 = 6*(i-1)
      xbase  = xbase  - y0(i61+1)*body(i)/body(0)
      ybase  = ybase  - y0(i61+2)*body(i)/body(0)
      zbase  = zbase  - y0(i61+3)*body(i)/body(0)
      vxbase = vxbase - y0(i61+4)*body(i)/body(0)
      vybase = vybase - y0(i61+5)*body(i)/body(0)
      vzbase = vzbase - y0(i61+6)*body(i)/body(0)
    end do
    do i = 1,ntot
      i61 = 6*(i-1)
      y(i61+1) = y0(i61+1) -  xbase
      y(i61+2) = y0(i61+2) -  ybase
      y(i61+3) = y0(i61+3) -  zbase
      y(i61+4) = y0(i61+4) - vxbase
      y(i61+5) = y0(i61+5) - vybase
      y(i61+6) = y0(i61+6) - vzbase
    end do
    xbase  = cero  ;  ybase  = cero  ;  zbase  = cero
    vxbase = cero  ;  vybase = cero  ;  vzbase = cero
  end if
  !
  if (idir == -1) then  ! convert astrocentric to barycentric
    !
    bodyt = body(0)
    do i = 1,npl
      i61 = 6*(i-1)
      bodyt = bodyt + body(i)
      xbase  = xbase  + y(i61+1)*body(i)
      ybase  = ybase  + y(i61+2)*body(i)
      zbase  = zbase  + y(i61+3)*body(i)
      vxbase = vxbase + y(i61+4)*body(i)
      vybase = vybase + y(i61+5)*body(i)
      vzbase = vzbase + y(i61+6)*body(i)
    end do
    xbase  =  -xbase/bodyt
    ybase  =  -ybase/bodyt
    zbase  =  -zbase/bodyt
    vxbase = -vxbase/bodyt
    vybase = -vybase/bodyt
    vzbase = -vzbase/bodyt
    do i = 1,ntot
      i61 = 6*(i-1)
      y0(i61+1) = y(i61+1) +  xbase
      y0(i61+2) = y(i61+2) +  ybase
      y0(i61+3) = y(i61+3) +  zbase
      y0(i61+4) = y(i61+4) + vxbase
      y0(i61+5) = y(i61+5) + vybase
      y0(i61+6) = y(i61+6) + vzbase
    end do
  end if
  !
end subroutine coord_bh

      
subroutine coord_ph (idir,y0,y) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y0(18*imax),y(18*imax),yi(18*imax)
  !
  vxbase = cero ; vybase = cero ; vzbase = cero
  ! 
  if (idir == 1) then   ! convert Poincare (y0) to astrocentric (y)
    y(1:6*ntot) = y0(1:6*ntot)
    do i = 1,npl
      vxbase = vxbase + y0(6*(i-1)+4)*body(i)
      vybase = vybase + y0(6*(i-1)+5)*body(i)
      vzbase = vzbase + y0(6*(i-1)+6)*body(i)
    end do
    vxbase = vxbase/body(0) ; vybase = vybase/body(0) ; vzbase = vzbase/body(0)
    do i = 1,ntot
      y(6*(i-1)+4) = y(6*(i-1)+4) + vxbase
      y(6*(i-1)+5) = y(6*(i-1)+5) + vybase
      y(6*(i-1)+6) = y(6*(i-1)+6) + vzbase
    end do
  end if
  !
  if (idir == -1) then  ! convert astrocentric (y) to Poincare (y0)
    y0(1:6*ntot) = y(1:6*ntot)
    bodytot = body(0)
    do i = 1,npl
      bodytot = bodytot + body(i)
      vxbase = vxbase + y(6*(i-1)+4)*body(i)
      vybase = vybase + y(6*(i-1)+5)*body(i)
      vzbase = vzbase + y(6*(i-1)+6)*body(i)
    end do
    vxbase = vxbase/bodytot ; vybase = vybase/bodytot ; vzbase = vzbase/bodytot
    do i = 1,ntot
      y0(6*(i-1)+4) = y0(6*(i-1)+4) - vxbase
      y0(6*(i-1)+5) = y0(6*(i-1)+5) - vybase
      y0(6*(i-1)+6) = y0(6*(i-1)+6) - vzbase
    end do
  end if
!
end subroutine coord_ph


subroutine coord_mh (idir,y0,y) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y0(18*imax),y(18*imax)
  !
  vxbase = cero  ;  vybase = cero  ;  vzbase = cero
  body_b = body(0) + body(1)
!!! 
  if (idir == 1) then   ! convert binary-Poincare (y0) to astrocentric (y)
    !
    do i = 2,npl             ! determine velocity of body(0) in new system
      beta_i = body(i)*body_b/(body(i) + body_b)
      vxbase = vxbase + y0(6*(i-1)+4)*beta_i
      vybase = vybase + y0(6*(i-1)+5)*beta_i
      vzbase = vzbase + y0(6*(i-1)+6)*beta_i
    end do
    vxbase = -(body(1)*y0(4) + vxbase)/body_b
    vybase = -(body(1)*y0(5) + vybase)/body_b
    vzbase = -(body(1)*y0(6) + vzbase)/body_b
    !
    y(1:6) = y0(1:6)
    do i = 2,ntot
      factor_v = body_b/(body_b+body(i))
      y(6*(i-1)+1) = y0(6*(i-1)+1) + y0(1)*body(1)/body_b
      y(6*(i-1)+2) = y0(6*(i-1)+2) + y0(2)*body(1)/body_b
      y(6*(i-1)+3) = y0(6*(i-1)+3) + y0(3)*body(1)/body_b
      y(6*(i-1)+4) = y0(6*(i-1)+4)*factor_v - vxbase
      y(6*(i-1)+5) = y0(6*(i-1)+5)*factor_v - vybase
      y(6*(i-1)+6) = y0(6*(i-1)+6)*factor_v - vzbase
    end do
  end if
  !
  if (idir == -1) then  ! convert astrocentric (y) to binary-poincare (y0)
    !
    bodyt = body(0)
    do i = 1,npl             ! determine astrocentric velocity of body(0)
      bodyt  = bodyt + body(i)
      vxbase = vxbase - y(6*(i-1)+4)*body(i)
      vybase = vybase - y(6*(i-1)+5)*body(i)
      vzbase = vzbase - y(6*(i-1)+6)*body(i)
    end do
    vxbase = vxbase/bodyt
    vybase = vybase/bodyt
    vzbase = vzbase/bodyt
    !
    y0(1:6) = y(1:6)
    do i = 2,ntot
      factor_v = (body_b+body(i))/body_b
      y0(6*(i-1)+1) =  y(6*(i-1)+1) - y(1)*body(1)/body_b
      y0(6*(i-1)+2) =  y(6*(i-1)+2) - y(2)*body(1)/body_b
      y0(6*(i-1)+3) =  y(6*(i-1)+3) - y(3)*body(1)/body_b
      y0(6*(i-1)+4) = (y(6*(i-1)+4) + vxbase)*factor_v
      y0(6*(i-1)+5) = (y(6*(i-1)+5) + vybase)*factor_v
      y0(6*(i-1)+6) = (y(6*(i-1)+6) + vzbase)*factor_v
    end do
  end if
!
end subroutine coord_mh
      
      
subroutine coord_planet_h (idir,ipla10,jmin,jmax,y,y0) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y0(18*imax),y(18*imax)
  common /cvs/ xbase,ybase,zbase,vxbase,vybase,vzbase
  !
  ipla = ipla10 - 10
!!! 
  if (idir == -1) then   ! convert astrocentric to planetocentric.
    i6p = 6*(ipla-1)
    do i = 1,ntot
      if (i /= ipla) then
        i61 = 6*(i-1)
        fac = cero
        if (i >= jmin .and. i <= jmax) fac = uno
        y0(i61+1) = y(i61+1) - y(i6p+1)*fac
        y0(i61+2) = y(i61+2) - y(i6p+2)*fac
        y0(i61+3) = y(i61+3) - y(i6p+3)*fac
        y0(i61+4) = y(i61+4) - y(i6p+4)*fac
        y0(i61+5) = y(i61+5) - y(i6p+5)*fac
        y0(i61+6) = y(i61+6) - y(i6p+6)*fac
      end if
    end do
  end if
  !
  if (idir == 1) then   ! convert planetocentric to astrocentric.
    i6p = 6*(ipla-1)
    do i = 1,ntot
      if (i /= ipla) then
        i61 = 6*(i-1)
        fac = cero
        if (i >= jmin .and. i <= jmax) fac = uno
        y0(i61+1) = y(i61+1) + y(i6p+1)*fac
        y0(i61+2) = y(i61+2) + y(i6p+2)*fac
        y0(i61+3) = y(i61+3) + y(i6p+3)*fac
        y0(i61+4) = y(i61+4) + y(i6p+4)*fac
        y0(i61+5) = y(i61+5) + y(i6p+5)*fac
        y0(i61+6) = y(i61+6) + y(i6p+6)*fac
      end if
    end do
  end if
  ! 
end subroutine coord_planet_h


subroutine aver (dm,e,u,f) ; use common
  implicit real*8 (a-h,k-z)
  !
  u0 = dm*rad
  dif = uno
  do while (dif > error) 
    u = dm*rad + e*sin(u0)
    dif = dabs(u - u0)
    u0 = u
  end do
  sen = sqrt(uno + e)*dsin(uno2*u) ; cos = sqrt(uno - e)*dcos(uno2*u)
  f   = dos*datan2(sen,cos)/rad    ; u   = u/rad
  !! 
end subroutine aver

      
subroutine per_sec (tau_sec) ; use common
  implicit real*8 (a-h,k-z)
  real*8 fac(0:20),b(0:20,9),m0,m1,m2
  !
  alfa1 = semi(npl-1)/semi(npl)
  alfa2 = alfa1*alfa1
  !
  do is = 1,9,2
    s = uno2*dfloat(is)
    fac(0) = uno
    do j = 0,20
      dj = dfloat(j)
      if (j > 0) fac(j) = fac(j-1)*((s+dj-uno)/dj)*alfa1
      di = 1.0
      factor = ((s+di-1.0)/di)*((s+di+dj-1.0)/(di+dj))
      b(j,is) = 1.0 + factor*alfa2
      i = 2
      dif = uno 
      do while (abs(dif) > error)
        di = dfloat(i)
        factor = factor*((s+di-uno)/di)*((s+di+dj-uno)/(di+dj))
        dif = factor*(alfa2**i)
        b(j,is) = b(j,is) + dif
        i = i + 1
      end do
      b(j,is) = 2.0*fac(j)*b(j,is)
    end do
  end do
  !
  A11 =  0.25*ene(npl-1)*(body(npl)/(body(0)+body(npl-1)))*alfa2*b(1,3)
  A22 =  0.25*ene(npl)*(body(npl-1)/(body(0)+body(npl)))*alfa1*b(1,3)
  A12 = -0.25*ene(npl-1)*(body(npl)/(body(0)+body(npl-1)))*alfa2*b(2,3)
  A21 = -0.25*ene(npl)*(body(npl-1)/(body(0)+body(npl)))*alfa1*b(2,3)
  !
  ca =  uno
  cb = -A11-A22
  cc =  A11*A22 - A21*A12
  raiz = cb*cb - 4.0*ca*cc
  !
  freq1 = uno2*(-cb + sqrt(raiz))
  freq2 = uno2*(-cb - sqrt(raiz))
  tau_sec = twopi/max(abs(freq1),abs(freq2))
  !
end subroutine per_sec


subroutine init_shannon (tau_sec) ; use common
  implicit real*8 (a-h,k-z)
  real*8 delta_a(imax),delta_e(imax),L2ini(imax),G2ini(imax),delta_G(imax)
  !
!!! numero de celdas.
  iqa_shan  = 3600                ! max = 2700
  iqe_shan  = 1600                ! max =  800
  qacc_shan = dfloat(iqa_shan*iqe_shan)
      
!!! region a muestrar para cada planeta en el plano (a,e).
  do i = 1,npl-1
    amed = uno2*(semi(i)+semi(i+1))
    Rh_mut = amed*((body(i)+body(i+1))/3.0/body(0))**(1.0/3.0)
    delta_a(i) = dos*sqrt(3.0)*Rh_mut ! limite de estabilidad de Marchal
    delta_e(i) = 0.5
  end do
  delta_a(npl) = delta_a(npl-1)
  delta_e(npl) = delta_e(npl-1)

!!! disminuye delta_a si este es mayor que la 1/2 distancia entre planetas.
  do i = 1,npl-1
    delmax = uno2*(semi(i+1)-semi(i))
    delta_a(i) = min(delta_a(i),delmax)
  end do
  delmax = uno2*(semi(npl)-semi(npl-1))
  delta_a(npl) = min(delta_a(npl),delmax)

!!! modifica delta_e al minimo valor para colision.
  do i = 1,npl-1
    alfai = semi(i)/semi(i+1)
    deltaei  = (uno/alfai)*(uno - exc(i+1)) - (uno + exc(i))
    deltaei1 = (uno - exc(i+1)) - alfai*(uno + exc(i))
    delta_e(i)   = min(delta_e(i),deltaei)
    delta_e(i+1) = min(delta_e(i+1),deltaei1)
  end do

!!! tamaño de la caja en (L²,G²).
  do i = 1,npl
    aamin = semi(i) - delta_a(i)
    aamax = semi(i) + delta_a(i)
    aamed = uno2*(aamin+aamax)
    eemin = max(cero,exc(i)-delta_e(i))
    eemax = min(uno,exc(i)+delta_e(i))
    eemed = uno2*(eemin+eemax)
    ! valores iniciales en acciones.
    L2ini(i) = semi(i)
    G2ini(i) = semi(i)*(uno-exc(i)**2)
    ! bordes de la caja en (L²,G²).
    L2boxmin(i) = L2ini(i) - delta_a(i)
    L2boxmax(i) = L2ini(i) + delta_a(i)
    G2boxmin(i) = L2boxmin(i)*(uno-eemed**2)
    G2boxmax(i) = L2boxmax(i)*(uno-eemin**2)
    delta_G(i)  = max(G2ini(i)-G2boxmin(i),G2boxmax(i)-G2ini(i))
    G2boxmin(i) = G2ini(i) - delta_G(i)
    G2boxmax(i) = G2ini(i) + delta_G(i)
    ! distancia cuadratica desde el centro (CI) a uno de los vertices.
    dist_box(i) = (L2boxmax(i)-L2ini(i))**2 + (G2boxmax(i)-G2ini(i))**2
  end do

!!! tiempo minimo para estimar entropia de Shannon.
!  tmin_shan = 5.0*tau_sec
  tmin_shan = cero

!!! Pablo's ad-hoc factor relating tau_esc with 1/D_S (still under discussion).
  K_S = 1.0

!!! inicializamos tiempos de escape.
  tesc_S(1:ntot) = 1.0d14
  !
end subroutine init_shannon

      
subroutine shannon (y) ; use common
  implicit real*8 (a-h,k-z)
  parameter (ipmax=33,iqamax=3700,iqemax=1800)
  real*8 y(18*imax),L2min(imax),L2max(imax),G2min(imax),G2max(imax)
  real*8 q0_acc(imax),P_acc(imax),S_acc(imax),D_S(imax),fac_sig(imax)
  real*8 nsum(imax),xsum(imax),ysumS(imax),x2sum(imax),xysumS(imax)
  real*8 yf(18*imax),Gsums(imax)
  integer ncel_acc(ipmax,iqamax,iqemax),num0_acc,num_acc
  save dsal,L2min,L2max,G2min,G2max,q0_acc,fac_sig,P_acc
  save nsum,xsum,ysumS,x2sum,xysumS
  save ncel_acc
  !
!!!! inicializamos cantidades.
  if (ifirst == 0) then
    ifirst  = 1     ;  dsal  = cero  ;  q0_acc = cero  ;  nsum   = cero
    L2min   = 1d10  ;  L2max =-1d10  ;  G2min  = 1d10  ;  G2max  =-1d10
    fac_sig = cero  ;  P_acc = cero  ;  S_acc  = cero  ;  D_S    = cero
    xsum    = cero  ;  ysumS = cero  ;  x2sum  = cero  ;  xysumS = cero
    ncel_acc = 0
  end if

!!! si hubo escape, setea tesc_S = tesc y no continua con el calculo.
  if (ifirst > 1) return
  if (npl /= npl0) then
    ifirst = 2 ; return
  end if
      
!!! numero de datos y su logaritmo.
  dsal = dsal + uno  ;  dlog_dsal = log(dsal)
      
!!! paso a vector de estado en Jacobi.
  call coord_jh (-1,yf,y)
  Gsums(1) = G*(body(0) + body(1))
  do i = 2,npl
    Gsums(i) = Gsums(i-1) + G*body(i)
  end do
  
!!!!! lazo sobre planetas.
  do 50 i = 1,npl

!!! semieje mayor y excentricidad (m0-centricos).
    i6 = 6*(i-1)
    hx = yf(i6+2)*yf(i6+6) - yf(i6+3)*yf(i6+5)
    hy = yf(i6+3)*yf(i6+4) - yf(i6+1)*yf(i6+6)
    hz = yf(i6+1)*yf(i6+5) - yf(i6+2)*yf(i6+4)
    h2 = hx*hx + hy*hy + hz*hz  
    r  = sqrt(yf(i6+1)*yf(i6+1) + yf(i6+2)*yf(i6+2) + yf(i6+3)*yf(i6+3))
    v2 = yf(i6+4)*yf(i6+4) + yf(i6+5)*yf(i6+5) + yf(i6+6)*yf(i6+6)
    energy = uno2*v2 - Gsums(i)/r
    semii = -uno2*Gsums(i)/energy
    e2 = uno - h2/(Gsums(i)*semii)
    
!!! valores asociados de L² & G².
    L2 = semii
    G2 = semii*(uno - e2)
    
!!! actualizamos valores maximos y minimos de L².
    L2min(i) = min(L2,L2min(i))  ;  L2max(i) = max(L2,L2max(i))
    G2min(i) = min(G2,G2min(i))  ;  G2max(i) = max(G2,G2max(i))
    
!!! actualiza factor sigma de Pablo.
    areamax = (L2max(i) - L2min(i))*(G2max(i) - G2min(i))
    fac_sig(i) = max(fac_sig(i),areamax)
    
!!! calculamos celda a la que corresponde cada variable.
    icel_a = int((L2-L2boxmin(i))*iqa_shan/(L2boxmax(i)-L2boxmin(i))) + 1
    icel_e = int((G2-G2boxmin(i))*iqe_shan/(G2boxmax(i)-G2boxmin(i))) + 1
       
!!! si alguna de las acciones supera limites permitidos, saca modulo.
    icel_a = mod(icel_a,iqa_shan)
    icel_e = mod(icel_e,iqe_shan)
    if (icel_a < 1) icel_a = icel_a + iqa_shan
    if (icel_e < 1) icel_e = icel_e + iqe_shan
    
!!! poblacion de la celda previa a este paso.
    num0_acc = ncel_acc(i,icel_a,icel_e)
    
!!! agregamos al contador en cada celda y actualizamos poblacion.
    ncel_acc(i,icel_a,icel_e) = ncel_acc(i,icel_a,icel_e) + 1
    num_acc = ncel_acc(i,icel_a,icel_e)
       
!!! numero de celdas recorridas.
    if (num0_acc == 0 .and. num_acc == 1) q0_acc(i) = q0_acc(i) + 1
    
!!! actualizamos suma de la entropia de Shannon.
    if (num0_acc > 0) then
      P_acc(i) = P_acc(i) - num0_acc*log(dfloat(num0_acc))
    end if
    P_acc(i) = P_acc(i) + num_acc*log(dfloat(num_acc))
    !     
50 end do
  
!!!! lazo para actualizacion de la entropia y coeficiente de difusion.
  do 80 i = 1,npl
    
!!! actualizamos valor de la entropia de Shannon.
    S_acc(i) = dlog_dsal - P_acc(i)/dsal
    
!!! estimamos tasa de difusion con dS/dt y con dq_0/dt.
    if (time > tmin_shan) then 
      tanio     = time/365.2563d0
      nsum(i)   = nsum(i)   + uno             ! N
      xsum(i)   = xsum(i)   + tanio           ! S_x
      ysumS(i)  = ysumS(i)  + S_acc(i)        ! S_y
      x2sum(i)  = x2sum(i)  + tanio**2        ! S_xx
      xysumS(i) = xysumS(i) + tanio*S_acc(i)  ! S_xy
      !
      dnumS     = nsum(i)*xysumS(i) - xsum(i)*ysumS(i)
      slopeS    = dnumS/(nsum(i)*x2sum(i) - xsum(i)**2)
      ord_oriS  = (ysumS(i) - slopeS*xsum(i))/nsum(i)
      dSdt_i    = slopeS
      D_S(i)    = fac_sig(i)*(q0_acc(i)/qacc_shan)*dSdt_i
    end if

!!! si time > tmin_shan, estima tiempo de escape proporcional a 1/abs(D_acc).
    if (time > tmin_shan) tesc_S(i) = K_S*dist_box(i)/abs(D_S(i))

80 end do
  !
end subroutine shannon


subroutine bs (y,t,delt,h) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax),dydx(18*imax)
  !
  htry = h ; tfinal = t + delt
  !
  do while (t < tfinal)
    if (t+htry > tfinal) htry = tfinal - t
    call force (t,y,dydx)
    call bstep (t,y,dydx,neq,htry,hdid,hnext)
    htry = hnext

!!! if requested, continue calculation of Shannon entropy.
    if (ishannon > 0) call shannon (y)
    
!!! check for collisions & escapes.
    if (icol(0) > 0) call collision (t,y)
    if (iesc(0) > 0) call escapes (1,t,y)
    ! 
  end do
  h = hnext
  !
end subroutine bs


subroutine normalize_lyap (y) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax)
  !
  dist0 = 1.0d-6
  if (maxval(icaos(1:6)).gt.0) then
    dist = cero
    do i = 1,npl
      ii = neq0 + neqs + 9*(i-1)
      dist = dist + y(ii+1)**2 + y(ii+2)**2 + y(ii+3)**2
      dist = dist + y(ii+4)**2 + y(ii+5)**2 + y(ii+6)**2
    end do
    dist = dsqrt(dist)
    do i = 1,npl
      ii = neq0 + neqs + 9*(i-1)
      y(ii+1) = y(ii+1)*dist0/dist
      y(ii+2) = y(ii+2)*dist0/dist
      y(ii+3) = y(ii+3)*dist0/dist
      y(ii+4) = y(ii+4)*dist0/dist
      y(ii+5) = y(ii+5)*dist0/dist
      y(ii+6) = y(ii+6)*dist0/dist
    end do
    do i = npl+1,ntot
      ii = neq0 + neqs + 9*(i-1)
      dist = y(ii+1)**2 + y(ii+2)**2 + y(ii+3)**2
      dist = dist + y(ii+4)**2 + y(ii+5)**2 + y(ii+6)**2
      dist = dsqrt(dist)
      y(ii+1) = y(ii+1)*dist0/dist
      y(ii+2) = y(ii+2)*dist0/dist
      y(ii+3) = y(ii+3)*dist0/dist
      y(ii+4) = y(ii+4)*dist0/dist
      y(ii+5) = y(ii+5)*dist0/dist
      y(ii+6) = y(ii+6)*dist0/dist
    end do
  end if
  !
end subroutine normalize_lyap
      

subroutine bstep (x,y,dydx,inv,htry,hdid,hnext) ; use common
  implicit real*8 (a-h,k-z)
  parameter (inmax=18*imax,ikmaxx=8)
  parameter (safe1=.25,safe2=.7,redmax=1.d-5,redmin=.7,tiny=1.d-30,scalmx=.1)
  integer inseq(ikmaxx+1)
  real*8 y(inmax),dydx(inmax),a(ikmaxx+1),alf(ikmaxx,ikmaxx)
  real*8 err(ikmaxx),yerr(inmax),ysav(inmax),yseq(inmax)
  logical first,reduct
  save a,alf,epsold,first,ikmax,ikopt,inseq,xnew
  data first /.true./ , epsold /-1./
  data uround /0.2220446049d-15/
  data inseq /2,4,6,8,10,12,14,16,18/
  !
  itermax = 10
  iter = 1
!!!
  if (eps /= epsold) then
    hnext = -1.0d29
    xnew  = -1.0d29
    eps1  = safe1*eps
    a(1)  = inseq(1) + 1
    do ik = 1,ikmaxx
      a(ik+1) = a(ik) + inseq(ik+1)
    end do
    do iq = 2,ikmaxx
      do ik = 1,iq-1
        alf(ik,iq) = eps1**((a(ik+1)-a(iq+1))/((a(iq+1)-a(1)+1.)*(2*ik+1)))
      end do
    end do
    epsold = eps
    do ikopt = 2,ikmaxx-1
      if (a(ikopt+1) > a(ikopt)*alf(ikopt-1,ikopt)) goto 1
    end do
1   ikmax = ikopt
  end if
  h = htry ; h = max(h,stepmin)
  do i = 1,inv
    ysav(i) = y(i)
  end do
  if (h /= hnext .or. x /= xnew) then
    first = .true. ; ikopt = ikmax
  end if
  reduct = .false.
2 continue
  iter = iter + 1
  do ik = 1,ikmax
    xnew = x + h
    if (xnew == x) write (*,*) 'step size underflow in bsstep'
    call mmid (ysav,dydx,inv,x,h,inseq(ik),yseq)
    xest = (h/inseq(ik))**2
    call pzextr (ik,xest,yseq,y,yerr,inv)
    if (ik /= 1) then
      errmax = tiny
      do i = 1,inv
        yscale = max(abs(y(i)),ascale)
        errmax = max(errmax,abs(yerr(i)/yscale))
      end do
      errmax = errmax/eps
      ikm = ik - 1
      err(ikm) = (errmax/safe1)**(1.0d0/dfloat(2*ikm+1))
    end if
    if (ik /= 1 .and. (ik >= ikopt-1 .or. first)) then
      if (errmax < 1) goto 4
      if (ik == ikmax .or. ik == ikopt+1) then
        red = safe2/err(ikm)
        goto 3
      else if (ik == ikopt) then
        if (alf(ikopt-1,ikopt) < err(ikm)) then
          red = 1.0d0/err(ikm)
          goto 3
        end if
      else if (ikopt == ikmax) then
        if (alf(ikm,ikmax-1) < err(ikm)) then
          red = alf(ikm,ikmax-1)*safe2/err(ikm)
          goto 3
        end if
      else if (alf(ikm,ikopt) < err(ikm)) then
        red = alf(ikm,ikopt-1)/err(ikm)
        goto 3
      end if
    end if
  end do
  !
3 red = min(red,redmin)
  red = max(red,redmax)
  h = h*red
  h = max(h,stepmin)
  reduct = .true.
  if (iter > itermax) return
  goto 2
4 x = xnew
  hdid = h
  first = .false.
  wrkmin = 1.d35
  do ikk = 1,ikm
    fact = max(err(ikk),scalmx)
    work = fact*a(ikk+1)
    if (work < wrkmin) then
      scale = fact
      wrkmin = work
      ikopt = ikk + 1
    end if
  end do
  hnext = h/scale
  if (ikopt >= ik .and. ikopt /= ikmax .and..not. reduct) then
    fact = max(scale/alf(ikopt-1,ikopt),scalmx)
    if (a(ikopt+1)*fact <= wrkmin) then
      hnext = h/fact ; ikopt = ikopt + 1
    end if
  end if
  !
end subroutine bstep


subroutine mmid (y,dydx,invar,xs,htot,instep,yout) ; use common
  implicit real*8 (a-h,k-z)
  parameter (inmax=18*imax)
  real*8 dydx(inmax),y(inmax),ym(inmax),yn(inmax),yout(inmax),derivs(inmax)
  !
  h  = htot/dfloat(instep)
  do i = 1,invar
    ym(i) = y(i)
    yn(i) = y(i) + h*dydx(i)
  end do
  x  = xs + h
  call force (x,yn,derivs)
  h2 = 2.0d0*h
  do in = 2,instep
    do i = 1,invar
      swap = ym(i) + h2*derivs(i)
      ym(i) = yn(i)
      yn(i) = swap
    end do
    x = x + h
    call force (x,yn,derivs)
  end do
  do i = 1,invar
    yout(i) = 0.5d0*(ym(i)+yn(i)+h*derivs(i))
  end do
  !
end subroutine mmid


subroutine pzextr (iest,xest,yest,yz,dy,inv) ; use common
  implicit real*8 (a-h,k-z)
  parameter (immax=23,inmax=18*imax)
  real*8 dy(inmax),yest(inmax),yz(inmax),d(inmax),qcol(inmax,immax),x(immax)
  save qcol,x
  ! 
  x(iest) = xest
  do j = 1,inv
    dy(j) = yest(j) ; yz(j) = yest(j)
  end do
  if (iest == 1) then
    do j = 1,inv
      qcol(j,1) = yest(j)
    end do
  else
    do j = 1,inv
      d(j) = yest(j)
    end do
    do ik1 = 1,iest-1
      delta = 1.0d0/(x(iest-ik1)-xest)
      f1 = xest*delta
      f2 = x(iest-ik1)*delta
      do j = 1,inv
        q = qcol(j,ik1)
        qcol(j,ik1) = dy(j)
        delta = d(j) - q
        dy(j) = f1*delta
        d(j)  = f2*delta
        yz(j) = yz(j) + dy(j)
      end do
    end do
    do j = 1,inv
      qcol(j,iest) = dy(j)
    end do
  end if
  ! 
end subroutine pzextr


subroutine radau (y,t,delt,h) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax)
  !
  call ra15 (y,t,delt,h,neq,1)
  
!!! check for collisions & escapes.
  if (iesc(0) > 0) call escapes (1,t,y)
  if (icol(0) > 0) call collision (t,y)
!
end subroutine radau


subroutine ra15 (x,t,tf,xl,nv,nclass) ; use common
  implicit real*8 (a-h,o-z)
  real*8 x(18*imax),f1(18*imax),fj(18*imax),c(21),d(21),r(21),y(18*imax)
  real*8 z(18*imax),b(7,18*imax),gra(7,18*imax),e(7,18*imax),bd(7,18*imax)
  real*8 h(8),w(7),u(7),nw(8)
  logical npq,nsf,nper,ncl,nes
  data nw /0,0,1,3,6,10,15,21/
  data zero,half,one,sr /0.0d0,0.5d0,1.0d0,1.4d0/
  data h /         0.d0,.05626256053692215d0,.18024069173689236d0, &
       .35262471711316964d0,.54715362633055538d0,.7342101772154105, &
       .88532094683909577d0,.97752061356128750d0/
  !
  t    = t + tf
  nper = .false.
  nsf  = .false.
  ncl  = nclass.eq.1
  npq  = nclass.lt.2
  !
  dir = one
  if (tf.lt.zero) dir = -one
  nes = ll.lt.0
  xl = dir*dabs(xl)
  pw = 1./9.
  !
  do 14 n = 2,8
    ww = n + n*n
    if (ncl) ww = n
    w(n-1) = one/ww
    ww = n
    u(n-1) = one/ww
14 end do
  do 22 k = 1,nv
    do 21 l = 1,7
      bd(l,k) = zero
      b(l,k) = zero
21  end do
22 end do
  w1 = half
  if (ncl) w1 = one
  c(1) =-h(2)
  d(1) = h(2)
  r(1) = one/(h(3)-h(2))
  la = 1
  lc = 1
  do 73 k = 3,7
    lb = la
    la = lc+1
    lc = nw(k+1)
    c(la) =-h(k)*c(lb)
    c(lc) = c(la-1) - h(k)
    d(la) = h(2)*d(lb)
    d(lc) =-c(lc)
    r(la) = one/(h(k+1)-h(2))
    r(lc) = one/(h(k+1)-h(k))
    if (k.eq.3) goto 73
    do 72 l = 4,k
      ld = la + l - 3
      le = lb + l - 4
      c(ld) = c(le) - h(k)*c(le+1)
      d(ld) = d(le) + h(l-1)*d(le+1)
      r(ld) = one/(h(k+1)-h(l-1))
72  end do
73 end do
  ss = 10.**(-ll)
  !
  tp = 0.1d0*dir
  if (xl.ne.zero) tp = xl
  if (nes) tp = xl
  if (tp/tf.gt.half) tp = half*tf
  ncount = 0
!!! line 4000 is the starting place of the first sequence.
4000 ns = 0
  nf = 0
  ni = 6
  tm = zero
  call force (t,x,f1)
  nf = nf + 1
  !
722 do 58 k = 1,nv
    gra(1,k) = b(1,k)+d(1)*b(2,k)+d(2)*b(3,k)+ &
         d(4)*b(4,k)+d(7)*b(5,k)+d(11)*b(6,k)+d(16)*b(7,k)
    gra(2,k) =             b(2,k)+d(3)*b(3,k)+ &
         d(5)*b(4,k)+d(8)*b(5,k)+d(12)*b(6,k)+d(17)*b(7,k)
    gra(3,k) = b(3,k)+d(6)*b(4,k)+d(9)*b(5,k)+d(13)*b(6,k)+d(18)*b(7,k)
    gra(4,k) =            b(4,k)+d(10)*b(5,k)+d(14)*b(6,k)+d(19)*b(7,k)
    gra(5,k) =                         b(5,k)+d(15)*b(6,k)+d(20)*b(7,k)
    gra(6,k) =                                      b(6,k)+d(21)*b(7,k)
    gra(7,k) =                                                   b(7,k)
58 end do
  tt = tp
  t2 = tt*tt
  if (ncl) t2 = tt
  tval = dabs(tt)
  !
  do 175 m = 1,ni
    do 174 j = 2,8
      jd = j - 1
      jdm = j - 2
      s = h(j)
      q = s
      if (ncl) q = one
      do 130 k = 1,nv
        a = w(3)*b(3,k)+s*(w(4)*b(4,k)+s*(w(5)*b(5,k)+s*(w(6)*b(6,k)+ &
             s*w(7)*b(7,k))))
        y(k) = x(k)+q*(t2*s*(f1(k)*w1+s*(w(1)*b(1,k)+s*(w(2)*b(2,k)+ &
             s*a))))
        if (npq) goto 130
        a = u(3)*b(3,k)+s*(u(4)*b(4,k)+s*(u(5)*b(5,k)+s*(u(6)*b(6,k)+ &
             s*u(7)*b(7,k))))
        z(k) = s*t*(f1(k)+s*(u(1)*b(1,k)+s*(u(2)*b(2,k)+s*a)))
130   end do
      !
      call force (t,y,fj)
      nf = nf + 1
      do 171 k = 1,nv
        temp = gra(jd,k)
        gk = (fj(k)-f1(k))/s
        goto (102,102,103,104,105,106,107,108),j
102     gra(1,k) = gk
        goto 160
103     gra(2,k) = (gk-gra(1,k))*r(1)
        goto 160
104     gra(3,k) = ((gk-gra(1,k))*r(2)-gra(2,k))*r(3)
        goto 160
105     gra(4,k) = (((gk-gra(1,k))*r(4)-gra(2,k))*r(5)-gra(3,k))*r(6)
        goto 160
106     gra(5,k) = ((((gk-gra(1,k))*r(7)-gra(2,k))*r(8)-gra(3,k))*r(9)- &
             gra(4,k))*r(10)
        goto 160
107     gra(6,k) = (((((gk-gra(1,k))*r(11)-gra(2,k))*r(12)-gra(3,k))*r(13)- &
             gra(4,k))*r(14)-gra(5,k))*r(15)
        goto 160
108     gra(7,k) = ((((((gk-gra(1,k))*r(16)-gra(2,k))*r(17)-gra(3,k))*r(18)- &
             gra(4,k))*r(19)-gra(5,k))*r(20)-gra(6,k))*r(21)
160     temp = gra(jd,k) - temp
        b(jd,k) = b(jd,k) + temp
        goto (171,171,203,204,205,206,207,208),j
203     b(1,k) = b(1,k) + c(1)*temp
        goto 171
204     b(1,k) = b(1,k) + c(2)*temp
        b(2,k) = b(2,k) + c(3)*temp
        goto 171
205     b(1,k) = b(1,k) + c(4)*temp
        b(2,k) = b(2,k) + c(5)*temp
        b(3,k) = b(3,k) + c(6)*temp
        goto 171
206     b(1,k) = b(1,k) + c(7)*temp
        b(2,k) = b(2,k) + c(8)*temp
        b(3,k) = b(3,k) + c(9)*temp
        b(4,k) = b(4,k) + c(10)*temp
        goto 171
207     b(1,k) = b(1,k) + c(11)*temp
        b(2,k) = b(2,k) + c(12)*temp
        b(3,k) = b(3,k) + c(13)*temp
        b(4,k) = b(4,k) + c(14)*temp
        b(5,k) = b(5,k) + c(15)*temp
        goto 171
208     b(1,k) = b(1,k) + c(16)*temp
        b(2,k) = b(2,k) + c(17)*temp
        b(3,k) = b(3,k) + c(18)*temp
        b(4,k) = b(4,k) + c(19)*temp
        b(5,k) = b(5,k) + c(20)*temp
        b(6,k) = b(6,k) + c(21)*temp
171   end do
174 end do
    if (nes.or.m.lt.ni) goto 175
!!! integration of sequence is over. next is sequence size control.
    hv = zero
    do 635 k = 1,nv
      hv = dmax1(hv,dabs(b(7,k)))
635 end do
    hv = hv*w(7)/tval/tval/tval/tval/tval/tval/tval
175 end do
  if (nsf) goto 180
  if (.not.nes) tp = (ss**pw)/(hv**pw)*dir
  if (nes) tp = xl
  if (nes) goto 170
  if (tp/tt.gt.one) goto 170
  tp = 0.8d0*tp
  ncount = ncount + 1
  if (ncount.gt.10) return
  goto 4000
170 nsf = .true.
180 do 35 k = 1,nv
    x(k) = x(k)+t2*(f1(k)*w1+b(1,k)*w(1)+b(2,k)*w(2)+b(3,k)*w(3) &
         + b(4,k)*w(4)+b(5,k)*w(5)+b(6,k)*w(6)+b(7,k)*w(7))
35 end do
  tm = tm + tt
  ns = ns + 1
  if (.not.nper) goto 78
  return
78 call force (t,x,f1)
  nf = nf + 1
  if (nes) goto 341
  tp = dir*(ss**pw)/(hv**pw)
  if (tp/t.gt.sr) tp = tt*sr
341 if (nes) tp = xl
  if (dir*(tm+tp).lt.dir*tf-1.d-8) goto 77
  tp = tf - tm
  nper = .true.
77 q = tp/tt
  do 39 k = 1,nv
    if (ns.eq.1) goto 31
    do 20 j = 1,7
      bd(j,k) = b(j,k) - e(j,k)
20  end do
31  e(1,k) =      q*(b(1,k)+ 2.d0*b(2,k)+ 3.d0*b(3,k)+ &
         4.d0*b(4,k)+ 5.d0*b(5,k)+ 6.d0*b(6,k)+ 7.d0*b(7,k))
    e(2,k) =                q**2*(b(2,k)+ 3.d0*b(3,k)+ &
         6.d0*b(4,k)+10.d0*b(5,k)+15.d0*b(6,k)+21.d0*b(7,k))
    e(3,k) =                             q**3*(b(3,k)+ &
         4.d0*b(4,k)+10.d0*b(5,k)+20.d0*b(6,k)+35.d0*b(7,k))
    e(4,k) =   q**4*(b(4,k)+ 5.d0*b(5,k)+15.d0*b(6,k)+35.d0*b(7,k))
    e(5,k) =                q**5*(b(5,k)+ 6.d0*b(6,k)+21.d0*b(7,k))
    e(6,k) =                             q**6*(b(6,k)+ 7.d0*b(7,k))
    e(7,k) =                                           q**7*b(7,k)
    do 38 l = 1,7
      b(l,k) = e(l,k) + bd(l,k)
38  end do
39 end do
!!! two iterations for every sequence after the first.
  ni = 2
  goto 722
  !
end subroutine ra15


subroutine rk (y,t,delt0,h) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax)
  !
  t_new = t + delt0 ; delt = delt0
1 continue
  call dopri8 (y,t,delt,h,neq)
  
!!! check for collisions & escapes.
  if (iesc(0) > 0) call escapes (1,t,y)
  if (icol(0) > 0) call collision (t,y)
  
!!! if collision or escape occured, finish integration interval.
  if (abs(t_new-t) > 1.0d-4*delt) then
    delt = t_new - t ; goto 1
  end if
  !
end subroutine rk


subroutine dopri8 (y,x,delt,h,in) ; use common
  implicit real*8 (a-h,k-z)
  real*8 k1(18*imax),k2(18*imax),k3(18*imax),k4(18*imax),k5(18*imax)
  real*8 k6(18*imax),k7(18*imax),y(18*imax),y1(18*imax)
  logical reject
  common /coef/ c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,a21,a31,a32,a41, &
       a43,a51,a53,a54,a61,a64,a65,a71,a74,a75,a76,a81,a84,a85,a86,a87,  &
       a91,a94,a95,a96,a97,a98,a101,a104,a105,a106,a107,a108,a109,a111,  &
       a114,a115,a116,a117,a118,a119,a1110,a121,a124,a125,a126,a127,a128,&
       a129,a1210,a1211,a131,a134,a135,a136,a137,a138,a139,a1310,a1311,  &
       b1,b6,b7,b8,b9,b10,b11,b12,b13,bh1,bh6,bh7,bh8,bh9,bh10,bh11,bh12
  data ini /0/ , inmax /150000/ , uround /0.2220446049d-15/
  !
  if (ini == 0) then
    ini = 1 ; call coefst
  end if
  !	 
  xend   = x + delt
  posneg = dsign(1.d0,xend-x)
  h      = dmin1(dmax1(1.d-10,dabs(h)),abs(stepmax))
  h      = dmax1(h,abs(stepmin)) ! set minimum possible time step
  h      = dsign(h,posneg)
  eps_do = dmax1(eps,uround)
  !
  reject = .false.
  inaccpt = 0 ; inrejct = 0 ; infcn = 0 ; instep = 0
1 if (instep > inmax .or. x+0.03d0*h == x) goto 79
  if ((x-xend)*posneg+uround > 0.d0) return
  if ((x+h-xend)*posneg > 0.d0) h = xend - x
  call force (x,y,k1)
2 continue
  instep = instep + 1
  y1(1:in) = y(1:in) + h*a21*k1(1:in)
  call force (x+c2*h,y1,k2)
  y1(1:in) = y(1:in) + h*(a31*k1(1:in)+a32*k2(1:in))
  call force (x+c3*h,y1,k3)
  y1(1:in) = y(1:in) + h*(a41*k1(1:in)+a43*k3(1:in))
  call force (x+c4*h,y1,k4)
  y1(1:in) = y(1:in) + h*(a51*k1(1:in)+a53*k3(1:in)+a54*k4(1:in))
  call force (x+c5*h,y1,k5)
  y1(1:in) = y(1:in) + h*(a61*k1(1:in)+a64*k4(1:in)+a65*k5(1:in))
  call force (x+c6*h,y1,k6)
  y1(1:in) = y(1:in) + h*(a71*k1(1:in)+a74*k4(1:in)+a75*k5(1:in)+a76*k6(1:in))
  call force (x+c7*h,y1,k7)
  y1(1:in) = y(1:in) + h*(a81*k1(1:in) &
       + a84*k4(1:in)+a85*k5(1:in)+a86*k6(1:in)+a87*k7(1:in))
  call force (x+c8*h,y1,k2)
  y1(1:in) = y(1:in) + h*(a91*k1(1:in) + a94*k4(1:in) &
       + a95*k5(1:in)+a96*k6(1:in)+a97*k7(1:in)+a98*k2(1:in))
  call force (x+c9*h,y1,k3)
  y1(1:in) = y(1:in) + h*(a101*k1(1:in)+a104*k4(1:in)+a105*k5(1:in) &
       + a106*k6(1:in)+a107*k7(1:in)+a108*k2(1:in)+a109*k3(1:in))
  do 61 i = 1,in
    y11s = a111*k1(i)+a114*k4(i)+a115*k5(i)+a116*k6(i)+a117*k7(i) &
         + a118*k2(i)+a119*k3(i)
    y12s = a121*k1(i)+a124*k4(i)+a125*k5(i)+a126*k6(i)+a127*k7(i) &
         + a128*k2(i)+a129*k3(i)
    k4(i) = a131*k1(i)+a134*k4(i)+a135*k5(i)+a136*k6(i)+a137*k7(i) &
         + a138*k2(i)+a139*k3(i)
    k5(i) = b1*k1(i) + b6*k6(i) + b7*k7(i) + b8*k2(i) + b9*k3(i)
    k6(i) = bh1*k1(i) + bh6*k6(i) + bh7*k7(i) + bh8*k2(i) + bh9*k3(i)
    k2(i) = y11s
    k3(i) = y12s
61 end do
  call force (x+c10*h,y1,k7)
  y1(1:in) = y(1:in) + h*(k2(1:in)+a1110*k7(1:in))
  call force (x+c11*h,y1,k2)
  xph = x + h
  y1(1:in) = y(1:in) + h*(k3(1:in)+a1210*k7(1:in)+a1211*k2(1:in))
  call force (xph,y1,k3)
  y1(1:in) = y(1:in) + h*(k4(1:in)+a1310*k7(1:in)+a1311*k2(1:in))
  call force (xph,y1,k4)
  infcn = infcn + 13
  do 35 i = 1,in
    k5(i) = y(i) + h*(k5(i)+b10*k7(i)+b11*k2(i)+b12*k3(i)+b13*k4(i))
    k6(i) = y(i) + h*(k6(i)+bh10*k7(i)+bh11*k2(i)+bh12*k3(i))
35 end do
  err = 0.d0
  do 41 i = 1,in
    denom = dmax1(1.d-6,dabs(k5(i)),dabs(y(i)),2.d0*uround/eps_do)
    err = err + ((k5(i)-k6(i))/denom)**2
41 end do
  err = dsqrt(err/dfloat(in))
  fac = dmax1((1.d0/6.d0),dmin1(3.d0,(err/eps_do)**(1.d0/8.d0)/.9d0))
  hnew = h/fac
  if (err > eps_do) goto 51
  inaccpt = inaccpt + 1
  y(1:in) = k5(1:in)
  x = xph
  if (dabs(hnew) > stepmax) hnew = posneg*stepmax
  if (reject) hnew = posneg*dmin1(dabs(hnew),dabs(h))
  reject = .false.
  h = hnew
  goto 1
51 reject = .true.
  h = hnew
  if (inaccpt >= 1) inrejct = inrejct + 1
  infcn = infcn - 1
  goto 2
79 continue
!
end subroutine dopri8


subroutine coefst
  implicit real*8 (a-h,k-z)
  common /coef/ c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13, &
       a21,a31,a32,a41,a43,a51,a53,a54,a61,a64,a65,a71,a74,a75,a76, &
       a81,a84,a85,a86,a87,a91,a94,a95,a96,a97,a98,a101,a104,a105,a106, &
       a107,a108,a109,a111,a114,a115,a116,a117,a118,a119,a1110,a121, &
       a124,a125,a126,a127,a128,a129,a1210,a1211,a131,a134,a135,a136, &
       a137,a138,a139,a1310,a1311,b1,b6,b7,b8,b9,b10,b11,b12,b13, &
       bh1,bh6,bh7,bh8,bh9,bh10,bh11,bh12 
  !
  c2    = 1.d0/18.d0                   ;  c3    = 1.d0/12.d0
  c4    = 1.d0/8.d0                    ;  c5    = 5.d0/16.d0
  c6    = 3.d0/8.d0                    ;  c7    = 59.d0/400.d0
  c8    = 93.d0/200.d0                 ;  c9    = 5490023248.d0/9719169821.d0
  c10   = 13.d0/20.d0                  ;  c11   = 1201146811.d0/1299019798.d0
  c12   = 1.d0                         ;  c13   = 1.d0
  a21   = c2                           ;  a31   = 1.d0/48.d0
  a32   = 1.d0/16.d0                   ;  a41   = 1.d0/32.d0
  a43   = 3.d0/32.d0                   ;  a51   = 5.d0/16.d0
  a53   =-75.d0/64.d0                  ;  a54   =-a53
  a61   = 3.d0/80.d0                   ;  a64   = 3.d0/16.d0
  a65   = 3.d0/20.d0                   ;  a71   = 29443841.d0/614563906.d0
  a74   = 77736538.d0/692538347.d0     ;  a75   =-28693883.d0/1125.d6
  a76   = 23124283.d0/18.d8            ;  a81   = 16016141.d0/946692911.d0
  a84   = 61564180.d0/158732637.d0     ;  a85   = 22789713.d0/633445777.d0
  a86   = 545815736.d0/2771057229.d0   ;  a87   =-180193667.d0/1043307555.d0
  a91   = 39632708.d0/573591083.d0     ;  a94   =-433636366.d0/683701615.d0
  a95   =-421739975.d0/2616292301.d0   ;  a96   = 100302831.d0/723423059.d0
  a97   = 790204164.d0/839813087.d0    ;  a98   = 800635310.d0/3783071287.d0
  a101  = 246121993.d0/1340847787.d0   ;  a104  =-37695042795.d0/15268766246.d0
  a105  =-309121744.d0/1061227803.d0   ;  a106  =-12992083.d0/490766935.d0
  a107  = 6005943493.d0/2108947869.d0  ;  a108  = 393006217.d0/1396673457.d0
  a109  = 123872331.d0/1001029789.d0   ;  a111  =-1028468189.d0/846180014.d0
  a114  = 8478235783.d0/508512852.d0   ;  a115  = 1311729495.d0/1432422823.d0
  a116  =-10304129995.d0/1701304382.d0 ;  a117  =-48777925059.d0/3047939560.d0
  a118  = 15336726248.d0/1032824649.d0 ;  a119  =-45442868181.d0/3398467696.d0
  a1110 = 3065993473.d0/597172653.d0   ;  a121  = 185892177.d0/718116043.d0
  a124  =-3185094517.d0/667107341.d0   ;  a125  =-477755414.d0/1098053517.d0
  a126  =-703635378.d0/230739211.d0    ;  a127  = 5731566787.d0/1027545527.d0
  a128  = 5232866602.d0/850066563.d0   ;  a129  =-4093664535.d0/808688257.d0
  a1210 = 3962137247.d0/1805957418.d0  ;  a1211 = 65686358.d0/487910083.d0
  a131  = 403863854.d0/491063109.d0    ;  a134  =-5068492393.d0/434740067.d0
  a135  =-411421997.d0/543043805.d0    ;  a136  = 652783627.d0/914296604.d0
  a137  = 11173962825.d0/925320556.d0  ;  a138  =-13158990841.d0/6184727034.d0
  a139  = 3936647629.d0/1978049680.d0  ;  a1310 =-160528059.d0/685178525.d0
  a1311 = 248638103.d0/1413531060.d0
  b1    = 14005451.d0/335480064.d0     ; b6    =-59238493.d0/1068277825.d0
  b7    = 181606767.d0/758867731.d0    ; b8    = 561292985.d0/797845732.d0
  b9    =-1041891430.d0/1371343529.d0  ; b10   = 760417239.d0/1151165299.d0
  b11   = 118820643.d0/751138087.d0    ; b12   =-528747749.d0/2220607170.d0
  b13   = 1.d0/4.d0
  bh1   = 13451932.d0/455176623.d0     ; bh6   =-808719846.d0/976000145.d0
  bh7   = 1757004468.d0/5645159321.d0  ; bh8   = 656045339.d0/265891186.d0
  bh9   =-3867574721.d0/1518517206.d0  ; bh10  = 465885868.d0/322736535.d0
  bh11  = 53011238.d0/667516719.d0     ; bh12  = 2.d0/45.d0
  !
end subroutine coefst


subroutine dump (io,y) ; use common
  implicit real*8 (a-h,k-z)
  real*8 y(18*imax)
  !
  if (io > 0) then         ! write dump file
    !
    open (44,file=ardp,status='replace',form='unformatted')
    !
    write (44) igenout,iplaout,iparout,iencout,ichaout,idmpout
    write (44) irel,itid,idrag,irtyp
    write (44) idu1,imig_stokes,imig_typ1,icav
    write (44) iyar,irelp,idec,idecs
    write (44) ista,im,ll,npl
    write (44) npart,ntot,neq0,neqs
    write (44) neqlyp,neq,iout
    write (44) iind,inty,iscr,ienel
    write (44) ietyp,ior,ios,ienc_first
    write (44) inout,inout0,npl0,npart0
    write (44) ntot0,ifil,ioout,icout
    do i = 1,16
      write (44) icaos(i)
    end do
    do i = 1,14
      write (44) iang(i)
    end do
    do i = 1,ntot0
      write (44) iesc(i),name(i),iorder(i)
    end do
    write (44) icol(0:2)
    !
    write (44) step,deltat,t0,tstop
    write (44) time,tout,sgn,unitt
    write (44) unitmp,ascale,eps
    write (44) unitm,unitd,stepmin,stepmax
    write (44) tdump,deltad,rmin2,rmax2
    write (44) semimin,semimax
    write (44) rhmin,demax,chaosmax
    write (44) egas,ggas,wgas0,rhop
    write (44) twopi,cero,uno,dos
    write (44) tres2,uno2,g,rad
    write (44) error,unoc2,Gsum0,t_stokes
    write (44) t_disk,sigma0,denpot,Hr0,ric
    write (44) delta_ic,Q_e,ang_fac,fac_tid
    write (44) unor2pi,fac_stokes,fac_mig
    do i = 1,10
      write (44) k2delt0(i),k2delti(i)
    end do
    do i = 0,10
      write (44) fac_zmi(i),zmi(i),love(i)
    end do
    do i = 0,2000
      write (44) airy(i)
    end do
    write (44) body(0),radius(0),rhill(0)
    write (44) rho(0),prot(0),dj2mod
    write (44) body0(0)
    do i = 1,ntot0
      write (44) body(i),radius(i),rhill(i),eta(i)
      write (44) body0(i),rho(i),prot(i)
      write (44) alfa(i),coefc(i),dadty(i),eleini(1,i)
      write (44) eleini(2,i),eleini(3,i),eleini(4,i)
      write (44) eleini(5,i),eleini(6,i),eleini(7,i)
      write (44) chaos(1,i),chaos(2,i),chaos(3,i),chaos(4,i)
      write (44) emin(i),dimin(i),eleini0(1,i)
      write (44) eleini0(2,i),eleini0(3,i),eleini0(4,i)
      write (44) eleini0(5,i),eleini0(6,i),eleini0(7,i)
      write (44) emax(i),dimax(i),amin(i),amax(i)
    end do
    do i = 1,18*ntot
      write (44) y(i)
    end do
    write (44) ardp
    write (44) arch_l
    write (44) arch
    write (44) arch_big
    write (44) arch_sma
    write (44) arch_enc
    write (44) archp_in
    write (44) archp_out
    do i = 1,ntot0
      write (44) arch_body(i)
    end do
!!!   
  else                      ! read dump file
    !
    open (44,file=ardp,status='old',form='unformatted')
    !
    read (44) igenout,iplaout,iparout,iencout,ichaout,idmpout
    read (44) irel,itid,idrag,irtyp
    read (44) idu1,imig_stokes,imig_typ1,icav
    read (44) iyar,irelp,idec,idecs
    read (44) ista,im,ll,npl
    read (44) npart,ntot,neq0,neqs
    read (44) neqlyp,neq,iout
    read (44) iind,inty,iscr,ienel
    read (44) ietyp,ior,ios,ienc_first
    read (44) inout,inout0,npl0,npart0
    read (44) ntot0,ifil,ioout,icout
    do i = 1,16
      read (44) icaos(i)
    end do
    do i = 1,14
      read (44) iang(i)
    end do
    iesc = 0
    name = 0
    do i = 1,ntot0
      read (44) iesc(i),name(i),iorder(i)
    end do
    read (44) icol(0:2)
    !
    read (44) step,deltat,t0,tstop
    read (44) time,tout,sgn,unitt
    read (44) unitmp,ascale,eps
    read (44) unitm,unitd,stepmin,stepmax
    read (44) tdump,deltad,rmin2,rmax2
    read (44) semimin,semimax
    read (44) rhmin,demax,chaosmax
    read (44) egas,ggas,wgas0,rhop
    read (44) twopi,cero,uno,dos
    read (44) tres2,uno2,g,rad
    read (44) error,unoc2,Gsum0,t_stokes
    read (44) t_disk,sigma0,denpot,Hr0,ric
    read (44) delta_ic,Q_e,ang_fac,fac_tid
    read (44) unor2pi,fac_stokes,fac_mig
    do i = 1,10
      read (44) k2delt0(i),k2delti(i)
    end do
    do i = 0,10
      read (44) fac_zmi(i),zmi(i),love(i)
    end do
    do i = 0,2000
      read (44) airy(i)
    end do
    body    = 0.0d0
    chaos   = 0.0d0
    amin    = 1.0d15
    amax    = 0.0d0
    emin    = 1.0d0
    emax    = 0.0d0
    dimin   = 6.28d0
    dimax   = 0.0d0
    eleini  = 0.0d0
    eleini0 = 0.0d0
    read (44) body(0),radius(0),rhill(0)
    read (44) rho(0),prot(0),dj2mod
    read (44) body0(0)
    do i = 1,ntot0
      read (44) body(i),radius(i),rhill(i),eta(i)
      read (44) body0(i),rho(i),prot(i)
      read (44) alfa(i),coefc(i),dadty(i),eleini(1,i)
      read (44) eleini(2,i),eleini(3,i),eleini(4,i)
      read (44) eleini(5,i),eleini(6,i),eleini(7,i)
      read (44) chaos(1,i),chaos(2,i),chaos(3,i),chaos(4,i)
      read (44) emin(i),dimin(i),eleini0(1,i)
      read (44) eleini0(2,i),eleini0(3,i),eleini0(4,i)
      read (44) eleini0(5,i),eleini0(6,i),eleini0(7,i)
      read (44) emax(i),dimax(i),amin(i),amax(i)
    end do
    do i = 1,18*ntot
      read (44) y(i)
    end do
    read (44) ardp
    read (44) arch_l
    read (44) arch
    read (44) arch_big
    read (44) arch_sma
    read (44) arch_enc
    read (44) archp_in
    read (44) archp_out
    do i = 1,ntot0
      read (44) arch_body(i)
    end do
!!! 
    idu1  = 0
    irtyp = 1
    t0    = time
    !
  end if
  !
  close (44)
  !
end subroutine dump


subroutine vocabulary (cartel,ncarteles)
  implicit real*8 (a-h,k-z)
  integer*4 ncarteles
  character(len=18) cartel(86)
  !
  cartel(1)  = 'new run or restart'   !    : new
  cartel(2)  = 'data dump file nam'   !    : dump.ncorp12
  cartel(3)  = 'data dump time int'   !    : 1.0d3
  !
  cartel(4)  = 'units of mass (sun'   !    : sun
  cartel(5)  = 'units of distance '   !    : au
  cartel(6)  = 'units of time (sec'   !    : year
  !
  cartel(7)  = 'integrator (radau,'   !    : bs
  cartel(8)  = 'initial time for i'   !    : 0.0
  cartel(9)  = 'total integration '   !    : 1.0d6
  cartel(10) = 'output time interv'   !    : 1.0d1
  cartel(11) = 'precision (digits '   !    : 11
  cartel(12) = 'initial time step '   !    : -1
  cartel(13) = 'sign (-1 for backw'   !    : 1.0
  !
  cartel(14) = 'mass of primary (i'   !    : 1.00
  cartel(15) = 'radius of primary '   !    : 590000.0
  cartel(16) = 'rotational period '   !    : 27.8
  cartel(76) = 'modified J2 (i.e. '   !    : 0.0
  !
  cartel(18) = 'include planets (m'   !    : yes
  cartel(17) = 'planet mass unit ('   !    : sun
  cartel(19) = 'planet ref. frame '   !    : astro
  cartel(77) = 'resonance indices '   !    : 
  cartel(23) = 'include Type I mig'   !    : no
  cartel(78) = 'eccentricity dampi'   !    : 0.5
  cartel(79) = 'ang. momentum cons'   !    : 0.3
  cartel(21) = 'include Stokes-typ'   !    : no
  cartel(22) = 'gas dissipation ti'   !    : 1.0d6

  cartel(31) = 'include tidal effe'   !    : no
  cartel(34) = 'stellar fluid Love'   !    : 0.38
  cartel(35) = 'stellar moment ine'   !    : 0.254
  cartel(81) = 'stellar tidal para'   !    : 1.0d6
  cartel(82) = 'planet fluid Love '   !    : 0.3  0.3  ...
  cartel(83) = 'planet moment iner'   !    : 0.33 0.33 ...
  cartel(84) = 'planetary tidal pa'   !    : 100. 100. ...
  cartel(32) = 'planet densities ['   !    : 1.0  1.0  ...
  cartel(33) = 'planet rotational '   !    : 0.3  0.3  ...
  cartel(86) = 'include star magne'   !    : no
  cartel(85) = 'acceleration facto'   !    : 1.0
  !
  cartel(36) = 'relativity for pla'   !    : no
  !
  cartel(37) = 'include particles '   !    : no
  cartel(38) = 'input particle fil'   !    : particles.in
  cartel(40) = 'include non-linear'   !    : no
  cartel(41) = 'planetesimal densi'   !    : 3.0
  cartel(48) = 'include Yarkovsky '   !    : no
  cartel(49) = 'albedo for particl'   !    : 0.04
  cartel(50) = 'relativity for par'   !    : no
  !
  cartel(24) = 'density at r=1 [gr'   !    : 1.0d5
  cartel(25) = 'power-law exponent'   !    : 0.5
  cartel(80) = 'disk flare index  '   !    : 0.0
  cartel(26) = 'disk scale height '   !    : 0.05
  cartel(27) = 'disk dissipation t'   !    : 1.0d6
  cartel(45) = 'eccentricity of ga'   !    : 0.0
  cartel(46) = 'initial disk long.'   !    : 0.0
  cartel(47) = 'inverse precession'   !    : 0.0
  cartel(28) = 'include inner cavi'   !    : no
  cartel(29) = 'location of cavity'   !    : 0.1
  cartel(30) = 'width of cavity tr'   !    : 0.01
  !
  cartel(20) = 'calculate indicato'   !    : no
  cartel(71) = 'maximum value for '   !    : 10.0
  !
  cartel(51) = 'minimum distance f'   !    : 0.001
  cartel(52) = 'maximum distance f'   !    : 100.0
  cartel(53) = 'minimum approach t'   !    : 0.1
  cartel(72) = 'minimum semimajor '   !    : 0.001
  cartel(73) = 'maximum semimajor '   !    : 100.0
  cartel(54) = 'maximum eccentrici'   !    : 0.9
  cartel(75) = 'stop run if planet'   !    : no
  !
  cartel(55) = 'use low-pass filte'   !    : no
  cartel(56) = 'decimation = int(t'   !    : 1
  cartel(57) = 'size of filter (ev'   !    : 200
  cartel(58) = 'output decimation '   !    : 1
  !
  cartel(59) = 'general output fil'   !    : ncorp12.dat
  cartel(60) = 'planetary data out'   !    : planets.dat
  cartel(61) = 'particles data out'   !    : particles.dat
  cartel(74) = 'collisions/escapes'   !    : encounters.dat
  cartel(62) = 'chaos indicator ou'   !    : chaos.dat
  cartel(63) = 'individual file pe'   !    : no
  cartel(64) = 'output on screen ('   !    : yes
  cartel(65) = 'output variables ('   !    : elements
  cartel(66) = 'output ref. frame '   !    : astro
  cartel(67) = 'output planet mass'   !    : no
  cartel(68) = 'output body radius'   !    : no
  cartel(69) = 'output stellar & p'   !    : no
  cartel(70) = 'output energy and '   !    : no
  !
  ncarteles = 86
  !
end subroutine vocabulary


subroutine option_char (ilmax,jbeg,cartel,rdata) ; use common
  implicit real*8 (a-h,k-z)
  integer*4 jbeg(300)
  character(len=18) cartel,command
  character(len=90) rdata
  !
  iflag = 0
  do i = 1,ilmax
    if (jbeg(i) /= 0) then
      command = lectura(i,jbeg(i))//lectura(i,jbeg(i)+1)//&
           lectura(i,jbeg(i)+2)//lectura(i,jbeg(i)+3)//&
           lectura(i,jbeg(i)+4)//lectura(i,jbeg(i)+5)//&
           lectura(i,jbeg(i)+6)//lectura(i,jbeg(i)+7)//&
           lectura(i,jbeg(i)+8)//lectura(i,jbeg(i)+9)//&
           lectura(i,jbeg(i)+10)//lectura(i,jbeg(i)+11)//&
           lectura(i,jbeg(i)+12)//lectura(i,jbeg(i)+13)//&
           lectura(i,jbeg(i)+14)//lectura(i,jbeg(i)+15)//&
           lectura(i,jbeg(i)+16)//lectura(i,jbeg(i)+17)
      if (command == cartel) then
        iflag = 1
        iflagp = 0
        j2p = jbeg(i) + 17
        do while (iflagp == 0) 
          if (lectura(i,j2p) == ':') iflagp = 1
          j2p = j2p + 1
        end do
        rdata = lectura(i,j2p)//lectura(i,j2p+1)//lectura(i,j2p+2)//&
             lectura(i,j2p+3)//lectura(i,j2p+4)//lectura(i,j2p+5)//&
             lectura(i,j2p+6)//lectura(i,j2p+7)//lectura(i,j2p+8)//&
             lectura(i,j2p+9)//lectura(i,j2p+10)//lectura(i,j2p+11)//&
             lectura(i,j2p+12)//lectura(i,j2p+13)//lectura(i,j2p+14)//&
             lectura(i,j2p+15)//lectura(i,j2p+16)//lectura(i,j2p+17)//&
             lectura(i,j2p+18)//lectura(i,j2p+19)//lectura(i,j2p+20)//&
             lectura(i,j2p+21)//lectura(i,j2p+22)//lectura(i,j2p+23)//&
             lectura(i,j2p+24)//lectura(i,j2p+25)//lectura(i,j2p+26)//&
             lectura(i,j2p+27)//lectura(i,j2p+28)//lectura(i,j2p+29)//&
             lectura(i,j2p+30)//lectura(i,j2p+31)//lectura(i,j2p+32)//&
             lectura(i,j2p+33)//lectura(i,j2p+34)//lectura(i,j2p+35)//&
             lectura(i,j2p+36)//lectura(i,j2p+37)//lectura(i,j2p+38)//&
             lectura(i,j2p+39)//lectura(i,j2p+40)//lectura(i,j2p+41)//&
             lectura(i,j2p+42)//lectura(i,j2p+43)//lectura(i,j2p+44)//&
             lectura(i,j2p+45)//lectura(i,j2p+46)//lectura(i,j2p+47)//&
             lectura(i,j2p+48)//lectura(i,j2p+49)//lectura(i,j2p+50)//&
             lectura(i,j2p+51)//lectura(i,j2p+52)//lectura(i,j2p+53)//&
             lectura(i,j2p+54)//lectura(i,j2p+55)//lectura(i,j2p+56)//&
             lectura(i,j2p+57)//lectura(i,j2p+58)//lectura(i,j2p+59)//&
             lectura(i,j2p+60)//lectura(i,j2p+61)//lectura(i,j2p+62)//&
             lectura(i,j2p+63)//lectura(i,j2p+64)//lectura(i,j2p+65)//&
             lectura(i,j2p+66)//lectura(i,j2p+67)//lectura(i,j2p+68)//&
             lectura(i,j2p+69)//lectura(i,j2p+70)//lectura(i,j2p+71)//&
             lectura(i,j2p+72)//lectura(i,j2p+73)//lectura(i,j2p+74)//&
             lectura(i,j2p+75)//lectura(i,j2p+76)//lectura(i,j2p+77)//&
             lectura(i,j2p+78)//lectura(i,j2p+79)//lectura(i,j2p+80)
      end if
    end if
  end do
  rdata = adjustl(rdata)
  !
end subroutine option_char


subroutine option_real (ilmax,jbeg,cartel,ddata) ; use common
  implicit real*8 (a-h,k-z)
  integer*4 jbeg(300)
  character(len=18) cartel,command
  character(len=90) rdata
  !
  iflag = 0
  do i = 1,ilmax
    if (jbeg(i) /= 0) then
      command = lectura(i,jbeg(i))//lectura(i,jbeg(i)+1)//&
           lectura(i,jbeg(i)+2)//lectura(i,jbeg(i)+3)//&
           lectura(i,jbeg(i)+4)//lectura(i,jbeg(i)+5)//&
           lectura(i,jbeg(i)+6)//lectura(i,jbeg(i)+7)//&
           lectura(i,jbeg(i)+8)//lectura(i,jbeg(i)+9)//&
           lectura(i,jbeg(i)+10)//lectura(i,jbeg(i)+11)//&
           lectura(i,jbeg(i)+12)//lectura(i,jbeg(i)+13)//&
           lectura(i,jbeg(i)+14)//lectura(i,jbeg(i)+15)//&
           lectura(i,jbeg(i)+16)//lectura(i,jbeg(i)+17)
      if (command == cartel) then
        iflag = 1
        iflagp = 0
        j2p = jbeg(i) + 17
        do while (iflagp == 0) 
          if (lectura(i,j2p) == ':') iflagp = 1
          j2p = j2p + 1
        end do
        rdata = lectura(i,j2p)
        do j = j2p+1,80
          rdata = trim(rdata)//lectura(i,j)
        end do
      end if
    end if
  end do
  rdata = trim(rdata)
  !
  if (iflag > 0) read (rdata,*) ddata
  !
end subroutine option_real


subroutine option_int (ilmax,jbeg,cartel,idata) ; use common
  implicit real*8 (a-h,k-z)
  integer*4 jbeg(300)
  character(len=18) cartel,command
  character(len=90) rdata
  !
  iflag = 0
  do i = 1,ilmax
    if (jbeg(i) /= 0) then
      command = lectura(i,jbeg(i))//lectura(i,jbeg(i)+1)//&
           lectura(i,jbeg(i)+2)//lectura(i,jbeg(i)+3)//&
           lectura(i,jbeg(i)+4)//lectura(i,jbeg(i)+5)//&
           lectura(i,jbeg(i)+6)//lectura(i,jbeg(i)+7)//&
           lectura(i,jbeg(i)+8)//lectura(i,jbeg(i)+9)//&
           lectura(i,jbeg(i)+10)//lectura(i,jbeg(i)+11)//&
           lectura(i,jbeg(i)+12)//lectura(i,jbeg(i)+13)//&
           lectura(i,jbeg(i)+14)//lectura(i,jbeg(i)+15)//&
           lectura(i,jbeg(i)+16)//lectura(i,jbeg(i)+17)
      if (command == cartel) then
        iflag = 1
        iflagp = 0
        j2p = jbeg(i) + 17
        do while (iflagp == 0) 
          if (lectura(i,j2p) == ':') iflagp = 1
          j2p = j2p + 1
        end do
        rdata = lectura(i,j2p)
        do j = j2p+1,80
          rdata = trim(rdata)//lectura(i,j)
        end do
      end if
    end if
  end do
  rdata = trim(rdata)
  !
  if (iflag > 0) read (rdata,*) idata
  !
end subroutine option_int


subroutine percentage (tout,tstop)
  implicit real*8 (a-h,k-z)
  integer iper
  character(len=1)  cret
  character(len=99) guiones
  character(len=3)  cestado
  save iper,guiones
  !
  cret = achar(13)          ! generate carriage return
!!! 
  iper = int(100.0*tout/tstop)
  guiones = ''
  do i = 1,iper
    guiones = trim(guiones)//'.'
  end do
!!! 
  open (66,status='scratch')
  if (iper.lt.10) then
    write (66,'(i2)') iper
  else
    write (66,'(i3)') iper
  end if
  rewind (66)
  read (66,'(a)') cestado
  close (66)
  ! 
  if (iper.lt.100) then
    write (*,110,advance='no') cret,trim(guiones)//trim(cestado)//'%'
  else
    write (*,110,advance='no') cret,trim(guiones)//'. FIN'
    write (*,*)
  end if
  ! 
110 format (2a)
  !
end subroutine percentage
