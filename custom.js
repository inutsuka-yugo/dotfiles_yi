require([
  'nbextensions/vim_binding/vim_binding',
], function() {
   CodeMirror.Vim.map("<C-a>", "ggVG", "normal");
});

require(["nbextensions/snippets_menu/main"], function (snippets_menu) {
    console.log('Loading `snippets_menu` customizations from `custom.js`');

    var horizontal_line = '---';
    var data_analysis = {
        "name": "data analysis",
        "sub-menu": [
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
            },
            {
                "name": "mkdir",
                "snippet": [
                    "os.makedirs(dir_name, exist_ok=True)"
                ]
            }
        ]
    };
    var my_favorites = {
        "name": "templates",
        "sub-menu": [
            {
                "name": "defaultdict",
                "snippet": [
                "from collections import defaultdict",
                "dic = defaultdict(int)",
                "",
                "for k, v in sorted(dic.items()):"
                ]
            },
            {
                "name": "Decimal",
                "snippet": [
                    "from decimal import *",
                    "x = Decimal(x)"
                ]
            },
            {
                "name": "deque",
                "snippet": [
                 "from collections import deque",
                 "# q = deque([0])",
                 "# while q:",
                 "#     x = q.pop()",
                 "#     q.append(1)",
                 "#     y = q.popleft()"
                ]
            },
            {
                "name": "priority queue",
                "snippet": [
                "from heapq import *",
                "",
                "hq = []",
                "heappush(hq, (1, x))",
                "t, x = heappop(hq)"
                ]
            },
            {
                "name": "accumulate",
                "snippet": [
                "from itertools import accumulate",
                "",
                "def accum_list(ls):",
                "    return list(accumulate(ls))"
                ]
            },
            {
                "name": "chmax, chmin for dp",
                "snippet": [
                "def chmin(dp, i, x): # dp[i] = min(dp[i], x)",
                "    if x<dp[i]:",
                "        dp[i]=x",
                "        return True",
                "    return False",
                " ",
                "def chmax(dp, i, x): # dp[i] = max(dp[i], x)",
                "    if x>dp[i]:",
                "        dp[i]=x",
                "        return True",
                "    return False"
                ]
            },
            {
                "name": "組み合わせ（順序なし）",
                "snippet": [
                    "from itertools import combinations",
                    "",
                    "list(combinations(range(5)))"
                ]
            },
            {
                "name": "順列（順序あり）",
                "snippet": [
                    "from itertools import permutations",
                    "",
                    "list(permutations(range(5)))"
                ]
            },
            {
                "name": "Combination, nCk",
                "snippet": [
                "# https://qiita.com/derodero24/items/91b6468e66923a87f39f",
                "from scipy.special import comb",
                "# a = comb(n, r)",
                "a = comb(n, r, exact=True)",
                "",
                "# nCk の max(n) が決まっていて、何度も計算する場合",
                "MOD = 998244353",
                "table_len = 301 # nCk の max(n)+1",
                "",
                "fac = [1, 1]",
                "for i in range(2, table_len):",
                "    fac.append(fac[-1] * i % MOD)",
                "",
                "finv = [0] * table_len",
                "finv[-1] = pow(fac[-1], MOD - 2, MOD)",
                "",
                "for i in range(table_len - 1, 0, -1):",
                "    finv[i - 1] = finv[i] * i % MOD",
                "",
                "",
                "def comb(n, k):",
                "    return fac[n] * finv[k] * finv[n - k] % MOD",
                "",
                "inv2 = pow(2, MOD-2, MOD)"
                ]
            },
            {
                "name": "b進数",
                "snippet": [
                "# base 10 --> base b (list)",
                "def shinsu(n, b):",
                "    ans = []",
                "    sho, amari = divmod(n, b)",
                "    ans.append(amari)",
                "    while sho > 0:",
                "        sho, amari = divmod(sho, b)",
                "        ans.append(amari)",
                "    return ans[::-1]",
                "",
                "# base b (list) --> base 10",
                "def bshinsu(l, b):",
                "    ans = 0",
                "    base = 1",
                "    for l1 in l[::-1]:",
                "        ans += l1 * base",
                "        base *= b",
                "    return ans"
                ]
            },
            {
                "name": "最大公約数、最小公倍数",
                "snippet": [
                "from math import gcd",
                "a,b=map(int, input().split())",
                "f=gcd(a,b) #最大公約数",
                "f2=a*b//f #最小公倍数"
                ]
            },
            {
                "name": "約数",
                "snippet": [
                "def divisors(M):",
                "    d = []",
                "    i = 1",
                "    while M >= i**2:",
                "        if M % i == 0:",
                "            d.append(i)",
                "            if i**2 != M:",
                "                d.append(M // i)",
                "        i = i + 1",
                "    return d"
                ]
            },
            {
                "name": "素数の列挙",
                "snippet": [
                "def prime_list(n): # list of prime numbers < n (by Sieve of Eratosthenes)",
                "    res = [0 for i in range(n + 1)]",
                "    prime = set([])",
                "    for i in range(2, n + 1):",
                "        if not res[i]:",
                "            prime.add(i)",
                "            for j in range(1, n // i + 1):",
                "                res[i * j] = 1",
                "    return prime"
                ]
            },
            {
                "name": "素因数分解→辞書へ",
                "snippet": [
                "def pf(m): # prime factorization",
                "    pfs = {}",
                "    for i in range(2, int(m**0.5) + 1):",
                "        while m % i == 0:",
                "            pfs[i] = pfs.get(i, 0) + 1",
                "            m //= i",
                "    if m > 1: pfs[m] = 1",
                "    return pfs",
                "",
                "pf(341555136)"
                ]
            },
            {
                "name": "modinv",
                "snippet": [
                "# https://tex2e.github.io/blog/crypto/modular-mul-inverse",
                "def xgcd(a, b):",
                "    x0, y0, x1, y1 = 1, 0, 0, 1",
                "    while b != 0:",
                "        q, a, b = a // b, b, a % b",
                "        x0, x1 = x1, x0 - q * x1",
                "        y0, y1 = y1, y0 - q * y1",
                "    return a, x0, y0",
                "",
                "def modinv(a, p): # a^-1 (mod p)",
                "    g, x, y = xgcd(a, p)",
                "    if g != 1:",
                "        raise Exception('modular inverse does not exist')",
                "    else:",
                "        return x % p",
                "",
                "",
                "# O(log m) but don't raise error",
                "def modinv(a, p): # a^-1 (mod p)",
                "    b, u, v = p, 1, 0",
                "    while b:",
                "        t = a // b",
                "        a -= t * b",
                "        a, b = b, a",
                "        u -= t * v",
                "        u, v = v, u",
                "    u %= p",
                "    if u < 0:",
                "        u += p",
                "    return u",
                "",
                "def modinv_list(n, p):",
                "    if n <= 1:",
                "        return [0, 1][:n + 1]",
                "    else:",
                "        inv_t = [0, 1]",
                "        for i in range(2, n + 1):",
                "            inv_t += [inv_t[p % i] * (p - int(p / i)) % p]",
                "        return inv_t"
                ]
            },
            {
                "name": "ax = b (mod p)",
                "snippet": [
                "from math import gcd",
                "",
                "def axbmodp(a, b, p): # a * x = b (mod p) ==> x [set, < p]",
                "    q = gcd(a, p)",
                "    a1 = a // q",
                "    p1 = p // q",
                "    b1, b2 = divmod(b, q)",
                "    if b2 != 0:",
                "        return set()",
                "    ret = set()",
                "    ans = (b1 * modinv(a1, p1)) % p1",
                "    while ans < p:",
                "        ret.add(ans)",
                "        ans += p1",
                "    return ret"
                ]
            },
            {
                "name": "merge_sort",
                "snippet": [
                "def merge_sort(A, B):",
                "    pos_A, pos_B = 0, 0",
                "    n, m = len(A), len(B)",
                "    res = []",
                "    while pos_A < n and pos_B < m:",
                "        a, b = A[pos_A], B[pos_B]",
                "        if a < b:",
                "            res.append(a)",
                "            pos_A += 1",
                "        else:",
                "            res.append(b)",
                "            pos_B += 1",
                "    res += A[pos_A:]",
                "    res += B[pos_B:]",
                "    return res"
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
    var lists = {
        "name": "lists",
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
                "name": "sort_index and ranking",
                "snippet": [
                    "def sort_index(ls):",
                    "        return sorted(range(len(ls)), key=lambda k: ls[k])",
                    "",
                    "def ranking(ls):",
                    "    return sort_index(sort_index(ls))",
                    "",
                    "a = [43, 91, 3535, 53]",
                    "a, sort_index(a), ranking(a)"
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
                "name": "transpose",
                "snippet": [
                "def transpose_tuple(l_2d):",
                "    return list(zip(*l_2d))",
                "",
                "def transpose(l_2d):",
                "    return [list(x) for x in zip(*l_2d)]"
                ]
            }
        ]
    }
    var compe_io = {
        "name": "I/O",
        "sub-menu": [
            {
                "name": "1行に複数の整数の入力",
                "snippet": " list(map(int, input().split()))"
            },
            {
                "name": "$l$行の整数の入力",
                "snippet": " [int(input()) for _ in range(l)]"
            },
            {
                "name": "$N$行の行列",
                "snippet": " [list(map(int, input().split())) for _ in range(N)]"
            },
            {
                "name": "faster input",
                "snippet": [
                "from sys import stdin",
                "input = stdin.readline"
                ]
            },
            {
                "name": "multiline output",
                "snippet": [
                    "print(*ans, sep='\\n')"
                ]
            }
        ]
    };
    var compe_basic = {
        "name": "Basic algorithms",
        "sub-menu": [
            {
                "name": "BFS on 2D map",
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
                    "    y,x = q.popleft()",
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
            },
            {
                "name": "BFS on 2D map w/ teleport",
                "snippet": [
                "Y, X = list(map(int, input().split()))",
                "",
                "from collections import deque",
                "",
                "tele = [[] for _ in range(26)]",
                "",
                "s = [list(input()) for _ in range(Y)] ",
                "",
                "orda = ord('a')",
                "",
                "for y in range(Y):",
                "    for x in range(X):",
                "        if s[y][x] == 'S':",
                "            sx, sy = x, y",
                "        elif s[y][x] == 'G':",
                "            gx, gy = x, y",
                "        elif not s[y][x] in '.#':",
                "            tele[ord(s[y][x]) - orda].append([y, x])",
                "",
                "inf = Y * X",
                "c = [[inf] * X for _ in range(Y)]",
                "",
                "c[sy][sx] = 0",
                "drs = [(-1, 0), (1, 0), (0, -1), (0, 1)]",
                "q = deque([(sy, sx)])",
                "",
                "while q:",
                "    y, x = q.popleft()",
                "    cost = c[y][x] + 1",
                "    for dx, dy in drs:",
                "        x2 = x + dx",
                "        y2 = y + dy",
                "        if x2 >= 0 and x2 < X and y2 >= 0 and y2 < Y and s[y2][x2] != '#':",
                "            if c[y2][x2] > cost:",
                "                c[y2][x2] = cost",
                "                q.append((y2, x2))",
                "    if s[y][x].islower():",
                "        for y3, x3 in tele[ord(s[y][x]) - orda]:",
                "            if c[y3][x3] > cost:",
                "                q.append((y3, x3))",
                "                c[y3][x3] = cost",
                "            tele[ord(s[y][x]) - orda] = []",
                "if c[gy][gx] == inf:",
                "    print(-1)",
                "else:",
                "    print(c[gy][gx])",
                "# for c1 in c:",
                "#     print(c1)",
                "# print(dic)"
                ]
            },
            {
                "name": "graph",
                "snippet": [
                "from collections import deque",
                "",
                "N = int(input())",
                "ab = [[0] * 2 for _ in range(N)]  # edge a -> b",
                "for i in range(N):",
                "    a, b = list(map(int, input().split()))",
                "    ab[i] = [a - 1, b - 1]",
                "nb = [set() for _ in range(N)]  # neighbor",
                "for a, b in ab:",
                "    nb[a].add(b)",
                "    # nb[b].add(a) # if bidirectional",
                "",
                "# DFS (Depth First Search)",
                "visited = {}",
                "q = deque([start])",
                "while q:",
                "    at = q.pop()",
                "    if at in visited:",
                "        continue",
                "    if at != start:",
                "        #####",
                "    visited.add(at)",
                "    for nb1 in nb[at]:",
                "        q.append(nb1)",
                "",
                "# BFS (Breadth First Search)",
                "visited = {}",
                "q = deque([start])",
                "while q:",
                "    at = q.popleft()",
                "    if at in visited:",
                "        continue",
                "    if at != start:",
                "        #####",
                "    visited.add(at)",
                "    for nb1 in nb[at]:",
                "        q.append(nb1)"
                ]
            },
            {
                "name": "tree",
                "snippet": [
                "from collections import deque",
                "",
                "N = int(input())",
                "ab = [[0] * 2 for _ in range(N - 1)]  # edge a -> b",
                "for i in range(N - 1):",
                "    a, b = list(map(int, input().split()))",
                "    ab[i] = [a - 1, b - 1]",
                "nb = [set() for _ in range(N)]  # neighbor",
                "for a, b in ab:",
                "    nb[a].add(b)",
                "    # nb[b].add(a) # if bidirectional",
                "",
                "# check depth",
                "depth = [-1] * N",
                "depth[0] = 0  # root is 0",
                "q = [0]",
                "while q:",
                "    at = q.pop()",
                "    for i in nb[at]:",
                "        if depth[i] == -1:",
                "            depth[i] = depth[at] + 1",
                "            q.append(i)",
                "",
                "# DFS (Depth First Search)",
                "q = deque([0])",
                "while q:",
                "    at = q.pop()",
                "    d = depth[at]",
                "    for to in nb[at]:",
                "        if depth[to] > d:  # go deeper and deeper (instead of checking visited)",
                "            q.append(to)",
                "            #####",
                "",
                "# DFS recursion",
                "def dfs(at, was):",
                "    for i in nb[at]:",
                "        if i == was:",
                "            continue",
                "        dfs(i, at)",
                "",
                "# BFS (Breadth First Search)",
                "q = deque([0])",
                "while q:",
                "    at = q.popleft()",
                "    d = depth[at]",
                "    for to in nb[at]:",
                "        if depth[to] > d:  # go deeper and deeper (instead of checking visited)",
                "            q.append(to)",
                "            #####"
                ]
            },
            {
                "name": "dijkstra",
                "snippet": [
                "from heapq import *",
                "INF = float('inf')",
                "",
                "def dijkstra(N, start, cost_nb):",
                "    hq = [(0, start)]",
                "    cost = [INF] * N",
                "    cost[start] = 0",
                "    ret = INF",
                "    while hq:",
                "        c, at = heappop(hq)",
                "        if c > cost[at]:",
                "            continue",
                "        for d, to in cost_nb[at]:",
                "            tmp = d + cost[at]",
                "            if to == start:",
                "                ret = min(ret, tmp)",
                "            if cost[to] > tmp:",
                "                cost[to] = tmp",
                "                heappush(hq, (tmp, to))",
                "    return ret",
                "",
                "",
                "def dijkstra_sg(N, start, goal, cost_nb):",
                "    hq = [(0, start)]",
                "    cost = [INF] * N",
                "    cost[start] = 0",
                "    while hq:",
                "        c, at = heappop(hq)",
                "        if c > cost[at]:",
                "            continue",
                "        for d, to in cost_nb[at]:",
                "            tmp = d + cost[at]",
                "            if cost[to] > tmp:",
                "                cost[to] = tmp",
                "                heappush(hq, (tmp, to))",
                "    return cost[goal]"
                ]
            },
            {
                "name": "巡回セールスマン",
                "snippet": [
                "INF = 1<<60",
                "dp = [[INF] * k for _ in range(1<<k)]",
                "for i in range(k):",
                "    dp[1<<i][i] = 1",
                "for s in range(1<<k):",
                "    for i in range(k):",
                "        for j in range(k):",
                "            if s>>i & 1:",
                "                if s>>j & 1:",
                "                    dp[s][j] = min(dp[s][j], dp[s^(1<<j)][i] + dist[i][j])"
                ]
            },
            {
                "name": "BIT",
                "snippet": [
                "class BIT:",
                "    def __init__(self, n, mod=0):  # 全要素数",
                "        self._n = n  # _n に全要素数を格納",
                "        self.data = [0] * n",
                "        self.mod = mod",
                "",
                "    def add(self, p, x):  # p は 0-indexed",
                "        assert 0 <= p < self._n",
                "        p += 1  # 0-indexed ==> 1-indexed",
                "        while p <= self._n:",
                "            self.data[p - 1] += x",
                "            if self.mod:",
                "                self.data[p - 1] %= self.mod",
                "            p += p & -p  # p & -p = LSB(p)",
                "",
                "    def _sum(self, r):",
                "        s = 0",
                "        while r > 0:",
                "            s += self.data[r - 1]",
                "            if self.mod:",
                "                s %= self.mod",
                "            r -= r & -r",
                "        return s",
                "",
                "    def sum(self, l, r):",
                "        assert 0 <= l <= r <= self._n",
                "        return self._sum(r) - self._sum(l)",
                ]
            },
            {
                "name": "BIT 2D",
                "snippet": [
                "# https://tjkendev.github.io/procon-library/python/range_query/bit.html",
                "class BIT2:",
                "    # H*W",
                "    def __init__(self, h, w):",
                "        self.w = w",
                "        self.h = h",
                "        self.data = [{} for i in range(h+1)]",
                "",
                "    # O(logH*logW)",
                "    def sum(self, i, j): # i,j は 1-indexed",
                "        s = 0",
                "        data = self.data",
                "        while i > 0:",
                "            el = data[i]",
                "            k = j",
                "            while k > 0:",
                "                s += el.get(k, 0)",
                "                k -= k & -k",
                "            i -= i & -i",
                "        return s",
                "",
                "    # O(logH*logW)",
                "    def add(self, i, j, x): # i,j は 1-indexed",
                "        w = self.w; h = self.h",
                "        data = self.data",
                "        while i <= h:",
                "            el = data[i]",
                "            k = j",
                "            while k <= w:",
                "                el[k] = el.get(k, 0) + x",
                "                k += k & -k",
                "            i += i & -i",
                "",
                "    # [x0, x1) x [y0, y1)",
                "    def range_sum(self, x0, x1, y0, y1): # 0-indexed !!!!!",
                "        return self.sum(x1, y1) - self.sum(x1, y0) - self.sum(x0, y1) + self.sum(x0, y0)",
                "",
                "    # handmade",
                "    def value(self, x, y): # 1-indexed",
                "        return self.range_sum(x-1, x, y-1, y)"
                ]
            },
            {
                "name": "FFT",
                "snippet": [
                "# Tallfall",
                "class NTT:",
                "    def __init__(self, MOD=998244353, pr=3, LS=20):",
                "        self.MOD = MOD",
                "        self.N0 = 1 << LS",
                "        omega = pow(pr, (MOD - 1) // self.N0, MOD)",
                "        omegainv = pow(omega, MOD - 2, MOD)",
                "        self.w = [0] * (self.N0 // 2)",
                "        self.winv = [0] * (self.N0 // 2)",
                "        self.w[0] = 1",
                "        self.winv[0] = 1",
                "        for i in range(1, self.N0 // 2):",
                "            self.w[i] = (self.w[i - 1] * omega) % MOD",
                "            self.winv[i] = (self.winv[i - 1] * omegainv) % MOD",
                "        used = set()",
                "        for i in range(self.N0 // 2):",
                "            if i in used:",
                "                continue",
                "            j = 0",
                "            for k in range(LS - 1):",
                "                j |= (i >> k & 1) << (LS - 2 - k)",
                "            used.add(j)",
                "            self.w[i], self.w[j] = self.w[j], self.w[i]",
                "            self.winv[i], self.winv[j] = self.winv[j], self.winv[i]",
                "",
                "    def _fft(self, A):",
                "        MOD = self.MOD",
                "        M = len(A)",
                "        bn = 1",
                "        hbs = M >> 1",
                "        while hbs:",
                "            for j in range(hbs):",
                "                A[j], A[j + hbs] = A[j] + A[j + hbs], A[j] - A[j + hbs]",
                "                if A[j] > MOD:",
                "                    A[j] -= MOD",
                "                if A[j + hbs] < 0:",
                "                    A[j + hbs] += MOD",
                "            for bi in range(1, bn):",
                "                wi = self.w[bi]",
                "                for j in range(bi * hbs * 2, bi * hbs * 2 + hbs):",
                "                    A[j], A[j + hbs] = (A[j] + wi * A[j + hbs]) % MOD, (",
                "                        A[j] - wi * A[j + hbs]) % MOD",
                "            bn <<= 1",
                "            hbs >>= 1",
                "",
                "    def _ifft(self, A):",
                "        MOD = self.MOD",
                "        M = len(A)",
                "        bn = M >> 1",
                "        hbs = 1",
                "        while bn:",
                "            for j in range(hbs):",
                "                A[j], A[j + hbs] = A[j] + A[j + hbs], A[j] - A[j + hbs]",
                "                if A[j] > MOD:",
                "                    A[j] -= MOD",
                "                if A[j + hbs] < 0:",
                "                    A[j + hbs] += MOD",
                "            for bi in range(1, bn):",
                "                winvi = self.winv[bi]",
                "                for j in range(bi * hbs * 2, bi * hbs * 2 + hbs):",
                "                    A[j], A[j + hbs] = (A[j] + A[j + hbs]) % MOD, winvi * (",
                "                        A[j] - A[j + hbs]) % MOD",
                "            bn >>= 1",
                "            hbs <<= 1",
                "",
                "    def convolve(self, A, B):",
                "        LA = len(A)",
                "        LB = len(B)",
                "        LC = LA + LB - 1",
                "        M = 1 << (LC - 1).bit_length()",
                "        A += [0] * (M - LA)",
                "        B += [0] * (M - LB)",
                "        self._fft(A)",
                "        self._fft(B)",
                "        C = [0] * M",
                "        for i in range(M):",
                "            C[i] = A[i] * B[i] % self.MOD",
                "        self._ifft(C)",
                "        minv = pow(M, self.MOD - 2, self.MOD)",
                "        for i in range(LC):",
                "            C[i] = C[i] * minv % self.MOD",
                "        return C[:LC]",
                "",
                "    def inverse(self, A):",
                "        LA = len(A)",
                "        dep = (LA - 1).bit_length()",
                "        M = 1 << dep",
                "        A += [0] * (M - LA)",
                "",
                "        g = [pow(A[0], self.MOD - 2, self.MOD)]",
                "        for n in range(dep):",
                "            dl = 1 << (n + 1)",
                "            f = A[:dl]",
                "            fg = self.convolve(f, g[:])[:dl]",
                "            fgg = self.convolve(fg, g[:])[:dl]",
                "            ng = [None] * dl",
                "            for i in range(dl // 2):",
                "                ng[i] = (2 * g[i] - fgg[i]) % self.MOD",
                "            for i in range(dl // 2, dl):",
                "                ng[i] = self.MOD - fgg[i]",
                "            g = ng[:]",
                "        return g[:LA]"
                ]
            }
        ]
    };

    // snippets_menu.default_menus[0]['sub-menu'].splice(3, 2); // Remove SymPy and pandas
    // snippets_menu.python.numpy['sub-menu-direction'] = 'left'; // Point new Numpy menus to left
    // snippets_menu.options['menus'].push(snippets_menu.default_menus[0]); // Start with the remaining "Snippets" menu
    // snippets_menu.options['menus'].push(snippets_menu.python.numpy); // Follow that with a new Numpy menu
    // snippets_menu.options['menus'].push(snippets_menu.python.scipy);
    // snippets_menu.options['menus'].push(snippets_menu.python.matplotlib);

    snippets_menu.options['menus'] = snippets_menu.default_menus;
    snippets_menu.options['menus'][0]['sub-menu'].push(horizontal_line);
    // snippets_menu.options['menus'].push(my_favorites);
    snippets_menu.options['menus'][0]['sub-menu'].push(my_favorites2);
    snippets_menu.options['menus'][0]['sub-menu'].push(my_favorites3);
    snippets_menu.options['menus'][0]['sub-menu'].push(my_favorites4);
    snippets_menu.options['menus'][0]['sub-menu'].push(data_analysis);
    snippets_menu.options['menus'].push(compe_io);
    snippets_menu.options['menus'].push(lists);
    snippets_menu.options['menus'].push(my_favorites);
    snippets_menu.options['menus'].push(compe_basic);
    console.log('Loaded `snippets_menu` customizations from `custom.js`');
});
