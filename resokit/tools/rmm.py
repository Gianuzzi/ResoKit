import numpy as np

G = 39.476926421373


def r3p(x, res):
    a, b, c = res
    r3p = -c / (a * x + b)

    # fix singularity
    sing = -b / a
    if np.ndim(x) > 0:
        if len(x) > 1:
            if sing >= np.min(x) and sing <= np.max(x):
                dif = np.abs(x - sing)
                minind = np.argmin(dif)
                r3p[minind] = np.nan
    return r3p


def rmm_in_area(lims, r3p_order=0, r3p_maxint=10, r2p=True, r2p_order=2, r2p_maxint=10):
    """
    Returns list of 3P-MMR as well as 2P-MMR in the
    x- and y- axes, respectively.
    """
    # ------------- 3P-MMR
    l1x, l2x, l1y, l2y = lims
    r3pl = []

    intl = np.flip(np.arange(-r3p_maxint, r3p_maxint + 1))

    for i in intl:
        i0 = i
        for j in intl:
            j0 = j
            for k in intl:
                i = i0
                j = j0

                if abs(i + j + k) > r3p_order:
                    continue

                if [i, j, k].count(0) > 1:
                    continue

                if (i == 0) or (k == 0):  # adjacent 2p-mmr
                    continue

                if j == 0:  # take (i,0,-k) over (-i,0,k)
                    if (i > 0) & (k < 0):
                        pass
                    else:
                        continue

                if (i < 0) & (j > 0) & (k < 0):
                    i *= -1
                    j *= -1
                    k *= -1

                fac = np.gcd.reduce([i, j, k])
                i //= fac
                j //= fac
                k //= fac

                if [i, j, k] in r3pl:
                    continue

                if (-j / i < l1x) or (-j / i > l2x):  # without singularities in domain
                    # si cruza el eje izquierdo
                    if (-k / (i * l1x + j) >= l1y) and (-k / (i * l1x + j) <= l2y):
                        pass

                    # si cruza el eje derecho
                    elif (-k / (i * l2x + j) >= l1y) and (-k / (i * l2x + j) <= l2y):
                        pass

                    # si cruza el eje de abajo
                    elif (-(j * l1y + k) / i / l1y >= l1x) and (
                        -(j * l1y + k) / i / l1y <= l2x
                    ):
                        pass

                    # no cruza ningun eje
                    else:
                        continue

                else:  # with singularities in domain
                    # si cruza el eje izquierdo
                    if (
                        (-j / i != l1x)
                        and (-k / (i * l1x + j) >= l1y)
                        and (-k / (i * l1x + j) <= l2y)
                    ):
                        pass

                    # si cruza el eje derecho
                    elif (
                        (-j / i != l2x)
                        and (-k / (i * l2x + j) >= l1y)
                        and (-k / (i * l2x + j) <= l2y)
                    ):
                        pass

                    # si cruza el eje de abajo
                    elif (
                        (-(j * l1y + k) / i / l1y >= l1x)
                        and (-(j * l1y + k) / i / l1y < -j / i)
                    ) or (
                        (-(j * l1y + k) / i / l1y > -j / i)
                        and (-(j * l1y + k) / i / l1y <= l2x)
                    ):
                        pass

                    # no cruza ningun eje
                    else:
                        continue

                r3pl.append([i, j, k])

    # ------------- 2P-MMR
    if r2p:
        r2px = []
        r2py = []

        intl = np.arange(1, r2p_maxint + 1)
        eq_ax = (l1y == l1x) & (l2y == l2x)
        for i in intl:
            i0 = i
            for j in intl:
                i = i0

                if j >= i:
                    continue
                if i - j > r2p_order:
                    continue

                fac = np.gcd.reduce([i, j])
                i //= fac
                j //= fac

                if (i / j >= l1x) & (i / j <= l2x):
                    if [i, j] not in r2px:
                        r2px.append([i, j])

                if not eq_ax:
                    if (i / j >= l1y) & (i / j <= l2y):
                        if [i, j] not in r2py:
                            r2py.append([i, j])
        if eq_ax:
            r2py = r2px

        return [r3pl, r2px, r2py]
    else:
        return r3pl


def r3p_label(res, ax, lims=False):
    a, b, c = res

    def r(x):
        r = -c / (a * x + b)
        return r

    def rinv(y):
        rinv = -(b * y + c) / a / y
        return rinv

    if lims:
        l1x, l2x, l1y, l2y = lims
    else:
        l1x, l2x = ax.get_xlim()
        l1y, l2y = ax.get_ylim()

    # cruza por eje derecho
    if (r(l2x) >= l1y) & (r(l2x) <= l2y):
        text = str(a) + " " + str(b) + " " + str(c)

        y = r(l2x)
        y_ax = (y - l1y) / (l2y - l1y)

        ax.text(1.01, y_ax, text, transform=ax.transAxes)

    # cruza por eje de arriba
    elif (rinv(l2y) >= l1x) & (rinv(l2y) < l2x):
        text = " " + str(a) + "\n" + str(b) + "\n " + str(c)

        x = rinv(l2y)
        x_ax = (x - l1x) / (l2x - l1x)
        ax.text(x_ax - 0.015, 1.02, text, transform=ax.transAxes)

    # no cruza por ninguno de esos
    else:
        print("Warning: " + str(res) + " does not cross right or top axis")
    return ()


def mindist_r3p(nnx, nny, mmr3p, return_coords=False, res=1e-6):
    """
    Calcula la distancia en el plano de razones de movimientos medios
    desde un punto (nnx,nny) a la resonancia mmr3p


    PARAMETERS
    ----------
    nnx,nny : float or arr of floats.
              Puntos en el espacio n_i/n_{i+1} , n_{i+1}/n_{i+2}. Pueden
              ser arrays de separaciones, en cuyo caso se devuelve una
              lista de distancias mínimas para cada par (nnx[i],nny[i])
    mmr3p   : lista de 3 enteros. E.g. mmr3p = [2,-5,3]. Es la resonancia
              de 3 cuerpos a la cual se calcula la distancia
    return_coords : Bool. Si True el programa devuelve (mindist, x, y) donde
                    (x,y) son los puntos sobre la curva 3P-MMR que corresponde
                    a la menor distancia hasta el punto (nnx,nny)
    res     : float or arr of floats.
              Resolución del método para encontrar la distancia minima. Puede
              ser un valor o un array de valores según la forma de nnx y nny.


    RETURNS
    -------
    Distancia minima del punto a la curva. Si return_coords es True, también
    devuelve las coordenadas sobre la curva correspondientes a la distancia
    minima
    """

    a, b, c = mmr3p

    if (
        len(np.shape(nnx)) == 1
        and len(np.shape(nny)) == 1
        and np.shape(nnx) == np.shape(nny)
    ):
        array = True
        N = len(nnx)
        mindistll = []
        x0l = []
        y0l = []
    elif len(np.shape(nnx)) == 0 and len(np.shape(nny)) == 0:
        array = False
        N = 1
    else:
        raise Exception("Fix shape of nnx and nny")

    for i in range(N):
        if array:
            nnxi = nnx[i]
            nnyi = nny[i]
        else:
            nnxi = nnx
            nnyi = nny
        x0 = nnxi - 0.5
        y0 = r3p(x0, [a, b, c])

        dx = 1
        xstuck = None

        singu = -b / a

        while abs(dx) > res:
            ac = a / c
            ac2 = ac**2.0
            ac3 = ac * ac2

            g0 = -2 * nnxi + 2 * x0 + 2 * y0**3 * ac - 2 * nnyi * y0**2 * ac
            d1g0 = 2 + 6 * ac2 * y0**4 - 4 * ac2 * nnyi * y0**3
            d2g0 = 24 * ac3 * y0**5 - 12 * nnyi * ac3 * y0**4

            dx = 2 * g0 * d1g0 / (2 * d1g0**2 - g0 * d2g0)

            x0 = x0 - dx
            y0 = r3p(x0, [a, b, c])

            # solo calculo la distancia al ala izquierda de la resonancia
            if (x0 >= singu) and (b != 0):
                if x0 == xstuck:
                    xstuck = "no_conv"
                    break
                if xstuck == "first_stuck_pass":
                    xstuck = x0
                if xstuck is None:
                    xstuck = "first_stuck_pass"
                x0 = singu - 5e-2
                y0 = r3p(x0, [a, b, c])
                dx = 1

        if xstuck == "no_conv":
            dom = np.linspace(1, singu - 1e-5, 1000000)
            mindistl = np.sqrt((nnxi - dom) ** 2 + (nnyi - r3p(dom, [a, b, c])) ** 2)
            argmin = np.argmin(mindistl)
            mindist = mindistl[argmin]
            x0 = dom[argmin]
            y0 = r3p(x0, [a, b, c])
        else:
            mindist = np.sqrt((nnxi - x0) ** 2 + (nnyi - y0) ** 2)

        if not array:
            if return_coords:
                return (mindist, x0, y0)
            else:
                return mindist

        mindistll.append(mindist)
        x0l.append(x0)
        y0l.append(y0)
    if return_coords:
        return (mindistll, x0l, y0l)
    else:
        return mindistll
