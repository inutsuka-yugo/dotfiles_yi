require([
  'nbextensions/vim_binding/vim_binding',
], function() {
   CodeMirror.Vim.map("<C-a>", "ggVG", "normal");
});
require(["nbextensions/snippets_menu/main"], function (snippets_menu) {
    console.log('Loading `snippets_menu` customizations from `custom.js`');
    var horizontal_line = '---';
    var my_favorites = {
        "name": "templates",
        "sub-menu": [
            {
                "name": "flatten",
                "snippet": [
                    "import collections",
                    "",
                    "def flatten(l):",
                    "    for el in l:",
                    "        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):",
                    "            yield from flatten(el)",
                    "        else:",
                    "            yield el"
                ]
            },
            {
                "name": "sort_index",
                "snippet": [
                    "def sort_index(ls):",
                    "        return sorted(range(len(ls)), key=lambda k: ls[k])"
                ]
            },
            {
                "name": "argmax",
                "snippet": [
                    "def argmax(ls):",
                    "    return ls.index(max(ls))"

                ]
            },
            {
                "name": "fitting histogram",
                "snippet": [
                    "histdata, bins = np.histogram(data, how_many_bins, [min, max], normed=True) # make histogram data",
                    "binscenters = (bins[1:] + bins[:-1])/2",
                    "inits = [1, 1, 1] # initial value",
                    "popt, pcov = curve_fit(multi_gauss, binscenters, histdata, p0=inits) # fitting",
                    "plt.bar(bins[:-1], histdata, width=1.0, alpha=0.6) # plot histogram"
                ]
            },
            {
                "name": "Gaussian",
                "snippet": [
                    "def gauss(x, mu, sigma, a):",
                    "    var = sigma ** 2",
                    "    return a * np.exp(-(x-mu)**2/2/var)",
                    "def gauss_list(x, *params):",
                    "    return [gauss(x, *params[i:i+3]) for i in range(0, len(params), 3)]",
                    "def multi_gauss(x, *params):",
                    "    return sum(gauss_list(x, *params))"
                ]
            },
            {
                "name": "multi-Gaussian plot",
                "snippet": [
                    "baseline = np.zeros_like(binscenters)",
                    "for gaussian in gauss_list(binscenters, *popt):",
                    "    plt.fill_between(binscenters, gaussian, baseline, alpha=0.3)",
                    "plt.legend()"
                ]
            },
            {
                "name": "calc MSD",
                "snippet": [
                    "def autocorrFFT(x):",
                    "  N=len(x)",
                    "  F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding",
                    "  PSD = F * F.conjugate()",
                    "  res = np.fft.ifft(PSD)",
                    "  res= (res[:N]).real   #now we have the autocorrelation in convention B",
                    "  n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)",
                    "  return res/n #this is the autocorrelation in convention A",
                    "  ",
                    "def msd_fft(r):",
                    "  N=len(r)",
                    "  D=np.square(r).sum(axis=1) ",
                    "  D=np.append(D,0) ",
                    "  S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])",
                    "  Q=2*D.sum()",
                    "  S1=np.zeros(N)",
                    "  for m in range(N):",
                    "      Q=Q-D[m-1]-D[N-m]",
                    "      S1[m]=Q/(N-m)",
                    "  return S1-2*S2"
                ]
            },
            {
                "name": "plot CDF",
                "snippet": [
                    "def cdf(array, **kwarg):",
                    "    n = len(array)",
                    "    cdfs = (np.arange(n)+1)/n",
                    "    plt.figure(**kwarg)",
                    "    plt.plot(np.sort(array, cdfs)"
                ]
            },
            {
                "name": "curve_fit",
                "snippet": [
                    "from scipy.optimize import curve_fit",
                    "",
                    "def fit(x, a, b):",
                    "    return a*x + b",
                    "",
                    "param, cov = curve_fit(fit, xs, ys)",
                    "ys_fit = fit(xs, *param)"
                ]
            }
        ]
    };
    var my_favorites2 = {
        "name": "iMinuit",
        "sub-menu": [
            {
                "name": "import",
                "snippet": [
                    "from iminuit import Minuit, describe",
                    "from iminuit.util import make_func_snippet",
                    "",
                    "class LeastSquares:  # override the class with a better one",
                    "    def __init__(self, model, x, y, err):",
                    "        self.model = model  # model predicts y for given x",
                    "        self.x = np.array(x)",
                    "        self.y = np.array(y)",
                    "        self.err = np.array(err)",
                    "        self.func_snippet = make_func_snippet(describe(self.model)[1:])",
                    "",
                    "    def __call__(self, *par):  # par are a variable number of model parameters",
                    "        ym = self.model(self.x, *par)",
                    "        chi2 = sum((self.y - ym)**2 / self.err)",
                    "        return chi2"
                ]
            },
            {
                "name": "fitting",
                "snippet": [
                    "lsq = LeastSquares(func_fit, x, y, err)",
                    "par_names = describe(func_fit)[1:]",
                    "lsq.func_snippet = make_func_snippet(par_names)",
                    "describe(lsq)",
                    "minuit = Minuit(lsq, a=0, error_a=0.1, errordef=1)",
                    "fmin, params = minuit.migrad()",
                    "pv = [param.value for param in params]"
                ]
            },
            {
                "name": "profile",
                "snippet": [
                    "minuit.draw_profile(param1)"
                ]
            }
        ]
    };
    var my_favorites3 = {
        "name": "pandas++",
        "sub-menu": [
            {
                "name": "read_csv in detail",
                "snippet": [
                    "df = pd.read_csv(",
                    "    './test.csv',",
                    "    sep=r',',",
                    "    skipinitialspace=True,",
                    "    header=None,",
                    "    names=[",
                    "        'datetime',",
                    "        'x',",
                    "    ],",
                    "    #index_col='datetime',",
                    "    parse_dates=[0]",
                    ")"
                ]
            },
            {
                "name": "original date_parser",
                "snippet": [
                    "# date_parser=my_parse",
                    "dtformat = '%Y%m%d%H%M%S.%f'",
                    "my_parser = lambda dt: pd.datetime.strptime(dt, dtformat)"
                ]
            }
        ]
    };
    var my_favorites4 = {
        "name": "ImageJ",
        "sub-menu": [
            {
                "name": "Split PTA data (folder `mW`)",
                "snippet": [
                "# split textfile to csvfiles",
                "filemin = 3",
                "",
                "os.makedirs('csv', exist_ok=True)",
                "",
                "for mW in mWs:",
                "    os.makedirs('csv/' + mW, exist_ok=True)",
                "    txtfiles = glob.glob(os.path.join(mW + \"/FIAllPoints\", '*.txt'))",
                "    print(txtfiles)",
                "    for txtfile in txtfiles:",
                "        dirname = os.path.basename(txtfile).replace(\"FIAllPoints.txt\", \"\")",
                "        csvpath = os.path.join('csv', mW, dirname)",
                "        print(csvpath)",
                "        os.makedirs(csvpath, exist_ok=True)",
                "        if glob.glob(os.path.join(",
                "                csvpath, '*.csv')) == []:  # if some csv exists, don't do this",
                "            f = open(txtfile, \"r\")",
                "            ls = []",
                "            lss = []",
                "            i = 0  # 今までに\"#\"を何回見たか",
                "            j = 0",
                "            for line in f:",
                "                if line[0] == \"#\":",
                "                    if i == 1:",
                "                        if ls != []:",
                "                            if j > filemin:",
                "                                lss.append(ls)",
                "                        ls = [line.replace('#', '')]",
                "                        i = 0",
                "                        j = 0",
                "                    else:",
                "                        i = 1",
                "                else:",
                "                    ls.append(line)",
                "                    j = j + 1",
                "            f.close()",
                "            lss.append(ls)",
                "",
                "            i = 0",
                "            for ls in lss:",
                "                name = csvpath + \"/\" + str(i) + \".csv\"",
                "                with open(name, mode='w') as f:",
                "                    for l in ls:",
                "                        f.write(l)",
                "                i = i + 1"
                ]
            },
            {
                "name": "Read data from splited csv",
                "snippet": [
                "framesss = []  # mW, experiment, frame length",
                "FIsss = []",
                "csvname = \"on_framess_length.csv\"",
                "",
                "for mW, csvdirs in zip(mWs, csvdirss):",
                "    csvpath = os.path.join(mW, csvname)",
                "",
                "    if os.path.exists(csvpath):",
                "        framess = pd.read_csv(csvpath)",
                "        if len(framess) > 1:",
                "            framess = framess.values",
                "            framess = [[",
                "                frame for frame in frames if np.logical_not(np.isnan(frame))",
                "            ] for frames in framess]",
                "        else:",
                "            framess = framess.to_numpy()",
                "    else:",
                "        framess = []",
                "        FIss = []",
                "        for csvdir in csvdirs:",
                "            files = glob.glob(os.path.join(csvdir, '*.csv'))",
                "            dfs = [pd.read_csv(file, sep=' ') for file in files]",
                "",
                "            frames = []",
                "            FIs = []",
                "            xs = []",
                "            ys = []",
                "",
                "            for df in dfs:",
                "                x = int(df['x'].mean())",
                "                y = int(df['y'].mean())",
                "                all_frames = df['Frame'].values",
                "                frames.append(all_frames[-1] - all_frames[0])",
                "                xs.append(x)",
                "                ys.append(y)",
                "                FIs.append(df[\"F.I.(I)\"].values.mean())",
                "            framess.append(frames)",
                "            FIss.append(FIs)",
                "            pd.DataFrame(framess).to_csv(csvpath, header=False, index=False)",
                "        pd.DataFrame(FIss).to_csv(csvpath.replace(\"framess\", \"FIss\"),",
                "                                  header=False,",
                "                                  index=False)",
                "    framesss.append(framess)",
                "    FIsss.append(FIss)"
                ]
            }
        ]
    };
    var my_favorites5 = {
        "name": "競プロ",
        "sub-menu": [
            {
                "name": "1行に複数の整数の入力",
                "snippet": "map(int, input().split())"
            },
            {
                "name": "$l$行の整数の入力",
                "snippet": [
                    "[int(input()) for _ in range(l)]"
                ]
            },
            {
                "name": "$N$行の行列",
                "snippet": [
                    "[list(map(int, input().split())) for _ in range(N)]"
                ]
            },
            {
                "name": "print",
                "snippet": [
                    "print(\"{}\".format(a))"
                ]
            },
            {
                "name": "素数",
                "snippet": [
                "n = 100",
                "primes = set(range(2, n+1))",
                "for i in range(2, int(n**0.5+1)):",
                "    primes.difference_update(range(i*2, n+1, i))",
                "primes=list(primes)"
                ]
            },
            {
                "name": "最大公約数、最小公倍数",
                "snippet": [
                "import fractions",
                "a,b=map(int, input().split())",
                "f=fractions.gcd(a,b) #最大公約数",
                "f2=a*b//f #最小公倍数"
                ]
            },
            {
                "name": "素因数分解→辞書へ",
                "snippet": [
                "pf={}",
                "m=341555136",
                "for i in range(2,int(m**0.5)+1):",
                "    while m%i==0:",
                "        pf[i]=pf.get(i,0)+1",
                "        m//=i",
                "if m>1:pf[m]=1",
                "print(pf)"
                ]
            },
            {
                "name": "BFS",
                "snippet": [
                    "from collections import deque",
                    "",
                    "Y, X = map(int,input().split())",
                    "sy, sx = map(int, input().split())",
                    "gy, gx = map(int, input().split())",
                    "s = [[\"#\"]*(X+2) for _ in range(Y+2)]",
                    "for y in range(Y):",
                    "    s[y+1] = list(\"#\"+input()+\"#\")",
                    "inf = Y*X",
                    "c = [[inf]*X for _ in range(Y)]",
                    "c[sy-1][sx-1] = 0",
                    "drs = [(-1,0),(1,0),(0,-1),(0,1)]",
                    "q = deque([(sy-1, sx-1)])",
                    "",
                    "while q:",
                    "    y,x = q.pop()",
                    "    cost = c[y][x]",
                    "    for dx,dy in drs:",
                    "        x2 = x+dx",
                    "        y2 = y+dy",
                    "        if s[y2+1][x2+1] == \".\" and c[y2][x2] > cost+1:",
                    "            c[y2][x2] = cost+1",
                    "            q.append((y2,x2))",
                    "if c[gy-1][gx-1] == inf:",
                    "    print(\"-1\")",
                    "else:",
                    "    print(c[gy-1][gx-1])"
                ]
            }
        ]
    };
    snippets_menu.options['menus'] = snippets_menu.default_menus;
    snippets_menu.options['menus'][0]['sub-menu'].push(horizontal_line);
    snippets_menu.options['menus'][0]['sub-menu'].push(my_favorites);
    snippets_menu.options['menus'][0]['sub-menu'].push(my_favorites2);
    snippets_menu.options['menus'][0]['sub-menu'].push(my_favorites3);
    snippets_menu.options['menus'][0]['sub-menu'].push(my_favorites4);
    snippets_menu.options['menus'][0]['sub-menu'].push(my_favorites5);
    console.log('Loaded `snippets_menu` customizations from `custom.js`');
});