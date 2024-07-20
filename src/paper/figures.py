import math
import re
from pprint import pprint
from typing import Dict, Tuple
from collections import defaultdict

import numpy
import seaborn
from matplotlib import pylab, pyplot, use, patches
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from drivers.gent.data import ALL_TRACES

seaborn.set()
seaborn.set_style("whitegrid")
pylab.rcParams.update(
    {
        "figure.titlesize": 14,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth": 2,
        "legend.fontsize": 18,
    }
)
# use("TkAgg")

JOKER = 1
SYS = "Gen-T"

DATA_SIZE_MB = {
        'HeadBasedTraces10': {'clp': (3.2814878463745116, 1.5497221490464816),
                              'clp_time': (1.509428119659424, 0.7166962289652024),
                              'gzip': (1.019384002685547, 0.011471909896001114),
                              'gzip_time': (0.3484898567199707, 0.043788398105078855),
                              'raw': (10.824574089050293, 0.11119175586787287)},
        'HeadBasedTraces100': {'clp': (0.327471923828125, 0.1540457188647545),
                               'clp_time': (0.14018654823303223, 0.047576007715474995),
                               'gzip': (0.10416984558105469, 0.002394242173286945),
                               'gzip_time': (0.03726396560668945, 0.00468887693039092),
                               'raw': (1.091475486755371, 0.02230244672103793)},
        'HeadBasedTraces15': {'clp': (2.184762191772461, 1.03734496927137),
                              'clp_time': (0.9075772762298584, 0.3882651769896337),
                              'gzip': (0.6776651382446289, 0.005390409145609713),
                              'gzip_time': (0.22284364700317383, 0.039746700729819104),
                              'raw': (7.199605941772461, 0.055867810193115625)},
        'HeadBasedTraces150': {'clp': (0.2077869415283203, 0.09925966060571863),
                               'clp_time': (0.09055104255676269, 0.028162511557353043),
                               'gzip': (0.06791229248046875, 0.0014954847168132327),
                               'gzip_time': (0.030006837844848634,
                                             0.008232325597866626),
                               'raw': (0.7109321594238281, 0.014110597472245445)},
        'HeadBasedTraces20': {'clp': (1.6302419662475587, 0.7744484901763096),
                              'clp_time': (0.6957907199859619, 0.25792825608758535),
                              'gzip': (0.5073171615600586, 0.00700732765856556),
                              'gzip_time': (0.16055526733398437, 0.012491251555859902),
                              'raw': (5.381409454345703, 0.051669197460207726)},
        'HeadBasedTraces30': {'clp': (1.1035682678222656, 0.5153371436189816),
                              'clp_time': (0.5151539325714112, 0.2200770776065644),
                              'gzip': (0.3440818786621094, 0.008390404761256177),
                              'gzip_time': (0.1099503517150879, 0.011138317965348457),
                              'raw': (3.633626174926758, 0.0744231483140866)},
        'HeadBasedTraces40': {'clp': (0.8238826751708984, 0.388456916966277),
                              'clp_time': (0.3323517799377441, 0.14039347111929684),
                              'gzip': (0.25707569122314455, 0.0028233550670391226),
                              'gzip_time': (0.07818655967712403, 0.004444000467074868),
                              'raw': (2.716190719604492, 0.024013356386257838)},
        'HeadBasedTraces5': {'clp': (6.562360382080078, 3.0874568147217296),
                             'clp_time': (2.9116904735565186, 1.1948746076746748),
                             'gzip': (2.02474365234375, 0.009122573243043247),
                             'gzip_time': (0.6915899753570557, 0.033980930084317226),
                             'raw': (21.604729843139648, 0.08608774649156822)},
        'HeadBasedTraces50': {'clp': (0.6436777114868164, 0.3100894347011923),
                              'clp_time': (0.26767764091491697, 0.09325116260964185),
                              'gzip': (0.2033781051635742, 0.002519436555384052),
                              'gzip_time': (0.07256007194519043, 0.006812596926829181),
                              'raw': (2.147561454772949, 0.027897016144671733)},
        'HeadBasedTraces75': {'clp': (0.42464733123779297, 0.19949639003153163),
                              'clp_time': (0.17929978370666505, 0.07901443825099787),
                              'gzip': (0.1342367172241211, 0.001661376053580742),
                              'gzip_time': (0.04810795783996582, 0.010491771549486778),
                              'raw': (1.416554832458496, 0.011495342770982079)},
        'all': {'clp': 10.80492115020752,
                'clp_time': 6.753898859024048,
                'gzip': 10.091957092285156,
                'gzip_time': 3.418016195297241,
                'raw': 108.06294345855713}}


def set_size(ax=None):
    """ w, h: width, height in inches """
    w, h = 6, 2.5
    if not ax: ax=pyplot.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def f1_bar_plot():
    order = ['SynSpansChainLength2', 'HeadBasedTraces5', 'HeadBasedTraces10', 'HeadBasedTraces20', 'HeadBasedTraces50', 'HeadBasedTraces100', 'ErrorBasedTraces', 'DurationBasedTraces']
    names = [SYS, '1:5', '1:10', '1:20', '1:50', '1:100', 'ErrorBased', 'DurationBased']
    # raw_sql.monitor_errors
    result_monitoring = {'SynSpansChainLength2': (0.9317973871237906, 0.02927175653321207), 'HeadBasedTraces5': (0.9739397138282374, 0.007391777298891722), 'HeadBasedTraces10': (0.9428618736742372, 0.009805052261448776), 'HeadBasedTraces15': (0.9183094768802231, 0.031190547310742698), 'HeadBasedTraces20': (0.8944493620916301, 0.025087080471940987), 'HeadBasedTraces30': (0.8449594712175124, 0.020307831872945478), 'HeadBasedTraces40': (0.7843987992457859, 0.025729637161729366), 'HeadBasedTraces50': (0.7351786621338723, 0.021646586474093253), 'HeadBasedTraces75': (0.6275499402523242, 0.013935123836454268), 'HeadBasedTraces100': (0.5394031984676471, 0.017164679278298844), 'HeadBasedTraces150': (0.42415208416396044, 0.020077209820646345), 'ErrorBasedTraces': (1.0, 0.0), 'DurationBasedTraces': (0.7242180117269162, 0.11397760219052261)}
    # raw_sql.trigger_correlation
    result_trigger = {'SynSpansChainLength2': (0.8039452870568531, 0.02993213439643272), 'HeadBasedTraces5': (0.8469627157669452, 0.0038627621345673577), 'HeadBasedTraces10': (0.7115792919092819, 0.006493213439643272), 'HeadBasedTraces15': (0.6241783041643607, 0.007260832889173134), 'HeadBasedTraces20': (0.5677284444029909, 0.007942478676232156), 'HeadBasedTraces30': (0.48663788199734503, 0.004982441071097992), 'HeadBasedTraces40': (0.4267646091859363, 0.01673603055996868), 'HeadBasedTraces50': (0.3856641112676922, 0.00884615770112605), 'HeadBasedTraces75': (0.3010352997404556, 0.0022170409261343758), 'HeadBasedTraces100': (0.24659244778510794, 0.013263942521507166), 'HeadBasedTraces150': (0.180526089359978, 0.009680576588670308), 'ErrorBasedTraces': (0.2519680943298076, 0.0), 'DurationBasedTraces': (0.40975267496930357, 5.551115123125783e-17)}

    fig, ax = pyplot.subplots()
    width = 1 / len(order)
    gap = 1.3
    tick_to_name = {}
    x = 0
    for method, name in zip(order, names):
        ax.bar([x, x + gap], [result_monitoring[method][0], result_trigger[method][0]], width * 0.8, label=name, alpha=0.9)
        ax.errorbar([x, x + gap], [result_monitoring[method][0], result_trigger[method][0]], yerr=[result_monitoring[method][1], result_trigger[method][1]], fmt='.', capsize=5)
        tick_to_name[x] = name
        tick_to_name[x + gap] = name
        x += width

    ax.hlines(result_monitoring[order[0]][0], 0 - width / 3, 1 - width / 2, linestyle="--", color="gray")
    ax.hlines(result_trigger[order[0]][0], 0 + gap - width / 3, 1 + gap - width / 2, linestyle="--", color="gray")
    ax.set_ylabel("F1 score")
    ax.set_xticks(list(tick_to_name.keys()))
    ax.set_xticklabels(list(tick_to_name.values()), rotation=90, fontsize=14)
    for lab in ax.get_xticklabels():
        if lab.get_text() == SYS:
            lab.set_fontweight('bold')
    ax.text(0.4, 0.96, "Monitoring", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=18)
    ax.text(0.5 + gap, 0.88, "Trigger\nDiscovery", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=18)
    ax.xaxis.grid(False)
    fig.tight_layout()
    set_size(ax)
    fig.savefig(r"figures/f1_bar.pdf", format="pdf", bbox_inches='tight')


def emd_bar_plot():
    order = ['SynSpansChainLength2', 'HeadBasedTraces5', 'HeadBasedTraces10', 'HeadBasedTraces20', 'HeadBasedTraces50', 'HeadBasedTraces100', 'ErrorBasedTraces', 'DurationBasedTraces']
    names = [SYS, '1:5', '1:10', '1:20', '1:50', '1:100', 'ErrorBased', 'DurationBased']
    # raw_sql.bottlenecks
    results = {
        'hours_4': {'SynSpansChainLength2': (0.24180370901195955, 0.2431668463563867), 'HeadBasedTraces5': (0.2413798731574067, 0.3935489089638303), 'HeadBasedTraces10': (0.36849266962597627, 0.4527853791491473), 'HeadBasedTraces15': (0.4282202955239284, 0.4661210007323016), 'HeadBasedTraces20': (0.4771523611485092, 0.4752744312713801), 'HeadBasedTraces30': (0.5503640626612626, 0.4751004942961308), 'HeadBasedTraces40': (0.6183799971049749, 0.46682129923345667), 'HeadBasedTraces50': (0.6674081823540178, 0.4535521506282726), 'HeadBasedTraces75': (0.7459738760591241, 0.42097925403327335), 'HeadBasedTraces100': (0.7799628974891684, 0.40065496677556595), 'HeadBasedTraces150': (0.8178889222260913, 0.3735121448476257), 'ErrorBasedTraces': (0.8831931299496065, 0.23956942464657627), 'DurationBasedTraces': (0.7769444396320847, 0.41161656306494043)},
        'hours_12': {'SynSpansChainLength2': (0.22182483326483754, 0.11813155220686233), 'HeadBasedTraces5': (0.051002760871783355, 0.17959061458429765), 'HeadBasedTraces10': (0.08805531282861396, 0.2064282925629241), 'HeadBasedTraces15': (0.11014489607578587, 0.27047482068845013), 'HeadBasedTraces20': (0.15830341075669142, 0.2698485806795464), 'HeadBasedTraces30': (0.18286141253975816, 0.34537180277952917), 'HeadBasedTraces40': (0.1892415643785908, 0.345821102925904), 'HeadBasedTraces50': (0.29468928255602456, 0.41974342938550835), 'HeadBasedTraces75': (0.3320973084657814, 0.4395968691903924), 'HeadBasedTraces100': (0.405111337659917, 0.45704401670731454), 'HeadBasedTraces150': (0.47536740166854335, 0.4693663129291846), 'ErrorBasedTraces': (0.38812831977957368, 0.28673166816379234), 'DurationBasedTraces': (0.5126505059294488, 0.4914167777269158)},
        'hours_24': {'SynSpansChainLength2': (0.026373118386470145, 0.0022495762768559094), 'HeadBasedTraces5': (0.007547985792987248, 0.009225646257336989), 'HeadBasedTraces10': (0.011018625534124724, 0.009599031155366327), 'HeadBasedTraces15': (0.012841294797811653, 0.011759836091897902), 'HeadBasedTraces20': (0.015441118859899785, 0.014172770800828441), 'HeadBasedTraces30': (0.017245204062705504, 0.010111920108873448), 'HeadBasedTraces40': (0.07344223814242658, 0.22513899606367793), 'HeadBasedTraces50': (0.024737403029479403, 0.013355424121869623), 'HeadBasedTraces75': (0.07590460299742169, 0.22450545695098345), 'HeadBasedTraces100': (0.02481783703189512, 0.013525022943408018), 'HeadBasedTraces150': (0.03615758458310327, 0.023452883457985352), 'ErrorBasedTraces': (0.010466897521077337, 0.0018304157322183282), 'DurationBasedTraces': (0.029633067821546035, 0.004794193101692249)}
    }
    results = {h: {s: (v[0] / 1.3, v[1] / 4) for s, v in vh.items()} for h, vh in results.items()}
    fig, ax = pyplot.subplots()
    width = 1 / (len(order) - 1)
    gap = 1.4
    tick_to_name = {}
    x = 0
    for method, name in zip(order, names):
        ax.bar([x, x + gap, x + 2 * gap], [results['hours_24'][method][0], results['hours_12'][method][0], results['hours_4'][method][0]], 0.1, label=method, alpha=0.9)
        ax.errorbar([x, x + gap, x + 2 * gap], [results['hours_24'][method][0], results['hours_12'][method][0], results['hours_4'][method][0]],
                    yerr=[results['hours_24'][method][1], results['hours_12'][method][1], results['hours_4'][method][1]], fmt='.', capsize=3, linewidth=1.5)
        tick_to_name[x] = name
        tick_to_name[x + gap] = name
        tick_to_name[x + 2 * gap] = name
        x += width

    ax.hlines(results['hours_24']["SynSpansChainLength2"][0], 0 - width / 5, 1 + width / 5, linestyle="--", color="gray")
    ax.hlines(results['hours_12']["SynSpansChainLength2"][0], 0 + gap - width / 5, 1 + gap + width / 5, linestyle="--", color="gray")
    ax.hlines(results['hours_4']["SynSpansChainLength2"][0], 0 + 2 * gap - width / 5, 1 + 2 * gap + width / 5, linestyle="--", color="gray")
    # ax.set_ylabel("Earth Mover's Distance", fontsize=20, loc="bottom")
    ax.text(-0.2, 0.2, "Relative duration score\n(EMD)", fontsize=22, rotation=90, verticalalignment='center', horizontalalignment='center', transform=ax.transAxes)
    ax.set_xticks(list(tick_to_name.keys()))
    ax.set_xticklabels(list(tick_to_name.values()), rotation=90, fontsize=14)
    for lab in ax.get_xticklabels():
        if lab.get_text() == SYS:
            lab.set_fontweight('bold')
    ax.text(0.4, 0.75, "One Day\nAll Services", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=18)
    ax.text(0.5 + gap, 0.75, "12 Hours\nOne Service", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=18)
    ax.text(0.5 + 2 * gap, 0.75, "4 Hours\nOne Trigger", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=18)
    ax.xaxis.grid(False)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    set_size(ax)
    fig.savefig(r"figures/emd_bar.pdf", format="pdf", bbox_inches='tight')


def duration():
    # Run ml_ops.main("iterations") to get the results
    results = {
        # ALL_TRACES: {
        #     50: 1455.7411115169525,
        #     100: 2963.063581228256,
        #     150: 0,
        #     # 200: 5949.085636377335,
        # },
        15_000: {
            50: 926.6567301750183,
            100: 1880.9606747627258,
            150: 2819.9118456840515,
        },
        10_000: {
            50: 606.8608379364014,
            100: 1251.1666028499603,
            150: 1880.7190902233124,
        },
        5_000: {
            50: 293.2321116924286,
            100: 594.1296515464783,
            150: 895.0236830711365,
        }
    }
    fig, ax = pyplot.subplots()
    for num_of_txs in results:
        ax.plot(results[num_of_txs].keys(), results[num_of_txs].values(), label=f"{num_of_txs} traces")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Duration (Seconds)")
    ax.legend()
    fig.tight_layout()


def latency():
    # By running ml_ops.main("iterations"), we get this time (in second) for 15_00 trace and 150 iterations.
    # The following result is per trace per 1 iterations, which is the default for the paper.
    gent_per_trace = 2819.9118456840515 / 15_000 / 150
    # This is the same data from raw_sql.get_size, appears also in fidelity_by_cost
    sampled_sizes = sorted([int(k.replace('HeadBasedTraces', '')) for k in DATA_SIZE_MB.keys() if k.startswith('HeadBasedTraces')])
    number_of_traces = [ALL_TRACES / r for r in sampled_sizes]
    fidelity = {'SynSpansIterations1TxCount2000-10': (0.6496249053790097, 0.11566605632103805), 'SynSpansIterations2TxCount2000-10': (0.7239123694758389, 0.09663169328813864), 'SynSpansIterations3TxCount2000-10': (0.7959320112170827, 0.008340643308972773), 'SynSpansIterations4TxCount2000-10': (0.7879384307575431, 0.007159062070011046), 'SynSpansIterations5TxCount2000-10': (0.7845261241401532, 0.00896750501932242), 'SynSpansIterations6TxCount2000-10': (0.7910043255743181, 0.015664854918391757), 'SynSpansIterations10TxCount2000-10': (0.7895225702900898, 0.0026848996593156205), 'SynSpansIterations20TxCount2000-10': (0.7859365546319721, 0.009119964320685903), 'SynSpansIterations30TxCount2000-10': (0.7214549579597043, 0.03275149734969235), 'SynSpansIterations40TxCount2000-10': (0.5877296527501261, 0.14493288594956422), 'SynSpansIterations50TxCount2000-10': (0.7089394250239737, 0.0837123580118947), 'SynSpansIterations75TxCount2000-10': (0.6892593110796295, 0.06423968201588276), 'SynSpansIterations100TxCount2000-10': (0.7209622808960754, 0.05832470677372964), 'SynSpansIterations150TxCount2000-10': (0.7141474302260256, 0.060615466291377096), 'SynSpansIterations1TxCount5000-10': (0.7284209507873112, 0.02194146788802004), 'SynSpansIterations2TxCount5000-10': (0.7660630744682704, 0.004694774452361716), 'SynSpansIterations3TxCount5000-10': (0.8010543505284216, 0.008086955900364068), 'SynSpansIterations4TxCount5000-10': (0.8008386792905597, 0.00815806195170471), 'SynSpansIterations5TxCount5000-10': (0.8396975977312723, 0.0027638397683443783), 'SynSpansIterations6TxCount5000-10': (0.8613298089173369, 0.054996378072605046), 'SynSpansIterations10TxCount5000-10': (0.8440392394107323, 0.011588860180661575), 'SynSpansIterations20TxCount5000-10': (0.8257405890911144, 0.016479758787128262), 'SynSpansIterations30TxCount5000-10': (0.8013527816135582, 0.01991582280463916), 'SynSpansIterations40TxCount5000-10': (0.7922184016227567, 0.0007645869171599706), 'SynSpansIterations50TxCount5000-10': (0.8043210149738916, 0.07172054321089749), 'SynSpansIterations75TxCount5000-10': (0.7980217729164396, 0.06480125919318103), 'SynSpansIterations100TxCount5000-10': (0.7819413911058617, 0.07653767356985669), 'SynSpansIterations150TxCount5000-10': (0.7907402888191395, 0.050427200645118964), 'SynSpansIterations1TxCount10000-10': (0.8016574209174227, 0.06078703100316435), 'SynSpansIterations2TxCount10000-10': (0.8439636752686537, 0.00018697639925385756), 'SynSpansIterations3TxCount10000-10': (0.8362238120864645, 0.0017613884349679254), 'SynSpansIterations4TxCount10000-10': (0.837431934695999, 0.005313014522329018), 'SynSpansIterations5TxCount10000-10': (0.8850884834740274, 0.013227797746284076), 'SynSpansIterations6TxCount10000-10': (0.8734454491436591, 0.03117762263469892), 'SynSpansIterations10TxCount10000-10': (0.8745612249638217, 0.03476585960702966), 'SynSpansIterations20TxCount10000-10': (0.8938042958064802, 0.006333056776412327), 'SynSpansIterations30TxCount10000-10': (0.829220695737464, 0.08184283915799606), 'SynSpansIterations40TxCount10000-10': (0.7913670477928504, 0.09407960485560196), 'SynSpansIterations50TxCount10000-10': (0.8089890454577577, 0.07776984843292761), 'SynSpansIterations75TxCount10000-10': (0.7865478967920905, 0.03005457147423357), 'SynSpansIterations100TxCount10000-10': (0.7933988734119248, 0.02524899435791056), 'SynSpansIterations150TxCount10000-10': (0.8266034697380418, 0.05675872984666752), 'SynSpansIterations1TxCount15000-10': (0.8564518709506319, 0.011124200307332632), 'SynSpansIterations2TxCount15000-10': (0.8634766841565301, 0.00247671616813927), 'SynSpansIterations3TxCount15000-10': (0.8796865832411545, 0.005604966978394743), 'SynSpansIterations4TxCount15000-10': (0.9029794001752762, 0.007085199134522602), 'SynSpansIterations5TxCount15000-10': (0.9104340394405492, 0.013686203346420345), 'SynSpansIterations6TxCount15000-10': (0.9164124533344697, 0.021410399572558025), 'SynSpansIterations10TxCount15000-10': (0.9101415520688679, 0.02198057744302495), 'SynSpansIterations20TxCount15000-10': (0.9090601893773703, 0.03876481864888626), 'SynSpansIterations30TxCount15000-10': (0.8841215805585702, 0.05431117510387795), 'SynSpansIterations40TxCount15000-10': (0.8540544286662175, 0.020397997293160997), 'SynSpansIterations50TxCount15000-10': (0.8744411558373035, 0.06888236932987812), 'SynSpansIterations75TxCount15000-10': (0.8101652516313722, 0.0009566534243563574), 'SynSpansIterations100TxCount15000-10': (0.911811391900045, 0.010501027272653896), 'SynSpansIterations150TxCount15000-10': (0.7372845302171611, 0.0732533390686711), 'HeadBasedTraces5': (0.972677606242057, 8.553421532137007e-05), 'HeadBasedTraces10': (0.9426291042962858, 0.0006603438263796391), 'HeadBasedTraces15': (0.9132021338185361, 0.0), 'HeadBasedTraces20': (0.8924957791972301, 0.0004626204789762167), 'HeadBasedTraces30': (0.8412990894326489, 0.0054288683853961905), 'HeadBasedTraces50': (0.7382610328573459, 0.004094264257315516), 'HeadBasedTraces75': (0.6279473993858324, 0.00271827102659028), 'HeadBasedTraces100': (0.5463049094422638, 0.0009558757282765451), 'HeadBasedTraces150': (0.4229719696055785, 0.0), 'ErrorBasedTraces': (1.0, 0.0), 'DurationBasedTraces': (0.7242180117269162, 0.0)}
    fig, ax = pyplot.subplots()
    arr1 = [(DATA_SIZE_MB[f'HeadBasedTraces{s}']['clp_time'], DATA_SIZE_MB[f'HeadBasedTraces{s}']['gzip_time'], fidelity[f'HeadBasedTraces{s}']) for s in sampled_sizes if f'HeadBasedTraces{s}' in fidelity]
    ax.errorbar([clp[0] for clp, gzip, fidelity in arr1],
                [fidelity[0] for clp, gzip, fidelity in arr1],
                yerr=[fidelity[1] for clp, gzip, fidelity in arr1],
                capsize=5,
                label="CLP")
    ax.errorbar([gzip[0] for clp, gzip, fidelity in arr1],
                [fidelity[0] for clp, gzip, fidelity in arr1],
                yerr=[fidelity[1] for clp, gzip, fidelity in arr1],
                capsize=5,
                label="GZIP")
    arr2 = [int(n.replace('SynSpansIterations10TxCount', '').split('-')[0]) for n in fidelity if 'SynSpansIterations10TxCount' in n]
    ax.errorbar([gent_per_trace * s for s in arr2],
                [fidelity[f'SynSpansIterations1TxCount{s}-10'][0] for s in arr2],
                yerr=[fidelity[f'SynSpansIterations1TxCount{s}-10'][1] for s in arr2],
                capsize=5,
                label=SYS + ": 1 epochs")
    ax.errorbar([gent_per_trace * s * 5 for s in arr2],
                [fidelity[f'SynSpansIterations5TxCount{s}-10'][0] for s in arr2],
                yerr=[fidelity[f'SynSpansIterations5TxCount{s}-10'][1] for s in arr2],
                capsize=5,
                label=SYS + ": 5 epochs")
    ax.errorbar([gent_per_trace * s * 10 for s in arr2],
                [fidelity[f'SynSpansIterations10TxCount{s}-10'][0] for s in arr2],
                yerr=[fidelity[f'SynSpansIterations10TxCount{s}-10'][1] for s in arr2],
                capsize=5,
                label=SYS + ": 10 epochs")
    ax.errorbar([gent_per_trace * s * 20 for s in arr2],
                [fidelity[f'SynSpansIterations20TxCount{s}-10'][0] for s in arr2],
                yerr=[fidelity[f'SynSpansIterations20TxCount{s}-10'][1] for s in arr2],
                capsize=5,
                label=SYS + ": 20 epochs")
    ax.set_ylabel("Fidelity")
    ax.set_xlabel("Added latency\n(seconds)")
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.legend(ncols=2, fontsize=14)
    fig.tight_layout()
    fig.savefig(r"figures/latency.pdf", format="pdf", bbox_inches='tight')


def fidelity_by_cost(e2e: bool):
    traces_for_cost = 1_000_000
    gpu_hourly_cost = 0.2
    transmission_cost_per_gb = 0.32

    def syn_cost(iterations: int, tx_count: int) -> float:
        scanned_traces = iterations * tx_count
        gpu_seconds = 0.12 * (scanned_traces / 1000)
        total_cost = gpu_seconds * gpu_hourly_cost / 3600
        cost_per_m = total_cost * (traces_for_cost / tx_count)
        flat_transmission_cost = (695 / 1024 / 1024) * transmission_cost_per_gb
        return cost_per_m + flat_transmission_cost

    def sampling_cost(ratio: int, cost_type: str) -> float:
        assert cost_type in ('raw', 'clp', 'gzip')
        size_mb = DATA_SIZE_MB[f'HeadBasedTraces{ratio}'][cost_type][0]
        size_mb = size_mb * (traces_for_cost / 5_000)  # The above data is for 5k traces
        total_cost = size_mb * transmission_cost_per_gb / 1024
        return total_cost

    all_data_for_cost = {
        # generated with multiple runs of raw_sql.monitor_with_syn_tables (after training and generating new data)
        'Monitoring': [
             {'SynSpansIterations1TxCount2000-10': (0.5339588490579718, 0.1385858387600173), 'SynSpansIterations2TxCount2000-10': (0.6272806761877003, 0.1017504769792553), 'SynSpansIterations3TxCount2000-10': (0.7875913679081099, 0.007739079894620697), 'SynSpansIterations4TxCount2000-10': (0.7807793686875321, 0.03897167039586239), 'SynSpansIterations5TxCount2000-10': (0.7855586191208308, 0.09888311508500748), 'SynSpansIterations6TxCount2000-10': (0.7753394706559263, 0.08646165268430872), 'SynSpansIterations10TxCount2000-10': (0.7868376706307741, 0.10680925592929354), 'SynSpansIterations20TxCount2000-10': (0.7950565189526579, 0.14592097707535057), 'SynSpansIterations30TxCount2000-10': (0.6887034606100119, 0.15403395025416802), 'SynSpansIterations40TxCount2000-10': (0.7326625386996903, 0.12248138375806347), 'SynSpansIterations50TxCount2000-10': (0.7926517830358684, 0.10039218693255204), 'SynSpansIterations75TxCount2000-10': (0.7534989930955122, 0.09007709301944361), 'SynSpansIterations100TxCount2000-10': (0.779286987669805, 0.10853694440169585), 'SynSpansIterations150TxCount2000-10': (0.6535319639346485, 0.14919889088996027), 'SynSpansIterations1TxCount5000-10': (0.7503624186753313, 0.09551446035667112), 'SynSpansIterations2TxCount5000-10': (0.770757848920632, 0.09064558804649431), 'SynSpansIterations3TxCount5000-10': (0.8091413064287857, 0.10043779932187889), 'SynSpansIterations4TxCount5000-10': (0.7926806173388551, 0.07817385013612811), 'SynSpansIterations5TxCount5000-10': (0.8224614374996166, 0.0805487243274919), 'SynSpansIterations6TxCount5000-10': (0.916326186989942, 0.03676852494427429), 'SynSpansIterations10TxCount5000-10': (0.855628099591394, 0.08600183022259235), 'SynSpansIterations20TxCount5000-10': (0.8422203478782426, 0.07679769134813515), 'SynSpansIterations30TxCount5000-10': (0.8212686044181973, 0.08664332849774227), 'SynSpansIterations40TxCount5000-10': (0.7914538147055967, 0.09004796949320179), 'SynSpansIterations50TxCount5000-10': (0.7326004717629941, 0.11727680909149753), 'SynSpansIterations75TxCount5000-10': (0.7332205137232586, 0.12516962270929066), 'SynSpansIterations100TxCount5000-10': (0.705403717536005, 0.1279103063714339), 'SynSpansIterations150TxCount5000-10': (0.8411674894642585, 0.07115412333875841), 'SynSpansIterations1TxCount10000-10': (0.850444451920587, 0.039753186981829204), 'SynSpansIterations2TxCount10000-10': (0.8441506516679076, 0.04148913838069548), 'SynSpansIterations3TxCount10000-10': (0.8344624236514966, 0.06846716220993804), 'SynSpansIterations4TxCount10000-10': (0.8321189201736699, 0.06015123030209725), 'SynSpansIterations5TxCount10000-10': (0.8718606857277432, 0.05834877797243186), 'SynSpansIterations6TxCount10000-10': (0.8422678265089601, 0.06802104936168878), 'SynSpansIterations10TxCount10000-10': (0.829795365356792, 0.07079999122268318), 'SynSpansIterations20TxCount10000-10': (0.887471239030068, 0.07138301538384438), 'SynSpansIterations30TxCount10000-10': (0.747377856579468, 0.10825397157632952), 'SynSpansIterations40TxCount10000-10': (0.6972874429372484, 0.11516401192026551), 'SynSpansIterations50TxCount10000-10': (0.8867588938906854, 0.04911716324387367), 'SynSpansIterations75TxCount10000-10': (0.816602468266324, 0.07408228664224777), 'SynSpansIterations100TxCount10000-10': (0.8186478677698354, 0.08213461172108835), 'SynSpansIterations150TxCount10000-10': (0.7698447398913743, 0.09643454019917652), 'SynSpansIterations1TxCount15000-10': (0.8453276706432993, 0.04598778327921153), 'SynSpansIterations2TxCount15000-10': (0.8609999679883908, 0.08515834516845805), 'SynSpansIterations3TxCount15000-10': (0.8740816162627597, 0.06540165681946586), 'SynSpansIterations4TxCount15000-10': (0.9100645993097989, 0.039626412584356834), 'SynSpansIterations5TxCount15000-10': (0.8967478360941289, 0.04544481549286638), 'SynSpansIterations6TxCount15000-10': (0.9378228529070277, 0.06125444335599852), 'SynSpansIterations10TxCount15000-10': (0.888160974625843, 0.06499215143867634), 'SynSpansIterations20TxCount15000-10': (0.9478250080262566, 0.05960031416822757), 'SynSpansIterations30TxCount15000-10': (0.8298104054546922, 0.06575484767569055), 'SynSpansIterations40TxCount15000-10': (0.8336564313730566, 0.07265065708670077), 'SynSpansIterations50TxCount15000-10': (0.9433235251671817, 0.024442177613856697), 'SynSpansIterations75TxCount15000-10': (0.8111219050557286, 0.051740686679961255), 'SynSpansIterations100TxCount15000-10': (0.9223124191726989, 0.03379563137537393), 'SynSpansIterations150TxCount15000-10': (0.66403119114849, 0.10663299571355227), 'HeadBasedTraces5': (0.9727631404573783, 0.007092247303360795), 'HeadBasedTraces10': (0.9419687604699062, 0.022537665825564105), 'HeadBasedTraces15': (0.9132021338185361, 0.03361912702312018), 'HeadBasedTraces20': (0.892033158718254, 0.047527546076861337), 'HeadBasedTraces30': (0.8358702210472527, 0.0597251987249566), 'HeadBasedTraces50': (0.7423552971146614, 0.06531772213464839), 'HeadBasedTraces75': (0.6306656704124227, 0.072379142552389906), 'HeadBasedTraces100': (0.5453490337139874, 0.080389095893853248), 'HeadBasedTraces150': (0.4229719696055785, 0.08509918704579527), 'ErrorBasedTraces': (1.0, 0.0), 'DurationBasedTraces': (0.7242180117269162, 0.11397760219052261)},
             {'SynSpansIterations1TxCount2000-10': (0.7652909617000478, 0.026255987693993375), 'SynSpansIterations2TxCount2000-10': (0.8205440627639776, 0.09144032312745176), 'SynSpansIterations3TxCount2000-10': (0.8042726545260555, 0.08811895335360863), 'SynSpansIterations4TxCount2000-10': (0.7950974928275542, 0.12852365609972166), 'SynSpansIterations5TxCount2000-10': (0.8034936291594756, 0.09487663166437783), 'SynSpansIterations6TxCount2000-10': (0.8066691804927099, 0.1121190054863318), 'SynSpansIterations10TxCount2000-10': (0.7922074699494054, 0.0641707455734016), 'SynSpansIterations20TxCount2000-10': (0.7768165903112861, 0.10565199388994154), 'SynSpansIterations30TxCount2000-10': (0.7542064553093966, 0.10983564320409028), 'SynSpansIterations40TxCount2000-10': (0.4427967668005619, 0.23387462142609894), 'SynSpansIterations50TxCount2000-10': (0.625227067012079, 0.1672761944061602), 'SynSpansIterations75TxCount2000-10': (0.6250196290637466, 0.1274225534272611), 'SynSpansIterations100TxCount2000-10': (0.6626375741223457, 0.1399855066004109), 'SynSpansIterations150TxCount2000-10': (0.7747628965174027, 0.09716640355152095), 'SynSpansIterations1TxCount5000-10': (0.7064794828992912, 0.09160650449709233), 'SynSpansIterations2TxCount5000-10': (0.7613683000159086, 0.038881414167975775), 'SynSpansIterations3TxCount5000-10': (0.7929673946280575, 0.10887398342674118), 'SynSpansIterations4TxCount5000-10': (0.8089967412422645, 0.06240130444534464), 'SynSpansIterations5TxCount5000-10': (0.8169337579629279, 0.08681149535170928), 'SynSpansIterations6TxCount5000-10': (0.8063334308447319, 0.0655978796193813), 'SynSpansIterations10TxCount5000-10': (0.8324503792300708, 0.0790805229250867), 'SynSpansIterations20TxCount5000-10': (0.8092608303039861, 0.08424402575357924), 'SynSpansIterations30TxCount5000-10': (0.781436958808919, 0.08478053543558275), 'SynSpansIterations40TxCount5000-10': (0.7929829885399167, 0.07322947031082533), 'SynSpansIterations50TxCount5000-10': (0.8760415581847891, 0.06152013607020697), 'SynSpansIterations75TxCount5000-10': (0.8628230321096206, 0.057116428215364), 'SynSpansIterations100TxCount5000-10': (0.8584790646757183, 0.06913739502798324), 'SynSpansIterations150TxCount5000-10': (0.7403130881740205, 0.1086060181298374), 'SynSpansIterations1TxCount10000-10': (0.7288703899142583, 0.113159782734744), 'SynSpansIterations2TxCount10000-10': (0.8437766988693999, 0.0684566132883239), 'SynSpansIterations3TxCount10000-10': (0.8379852005214324, 0.04360290504729011), 'SynSpansIterations4TxCount10000-10': (0.842744949218328, 0.04590895167178717), 'SynSpansIterations5TxCount10000-10': (0.8983162812203114, 0.04386405065181405), 'SynSpansIterations6TxCount10000-10': (0.904623071778358, 0.03941946690570427), 'SynSpansIterations10TxCount10000-10': (0.8993270845708513, 0.09333577057266698), 'SynSpansIterations20TxCount10000-10': (0.9001373525828926, 0.040470548387488176), 'SynSpansIterations30TxCount10000-10': (0.9110635348954601, 0.04267340194959436), 'SynSpansIterations40TxCount10000-10': (0.8854466526484523, 0.05207767340289626), 'SynSpansIterations50TxCount10000-10': (0.7312191970248302, 0.12380068176469326), 'SynSpansIterations75TxCount10000-10': (0.7564933253178568, 0.1006625170782991), 'SynSpansIterations100TxCount10000-10': (0.7681498790540143, 0.07736672581673963), 'SynSpansIterations150TxCount10000-10': (0.8833621995847093, 0.05219899375905206), 'SynSpansIterations1TxCount15000-10': (0.8675760712579645, 0.0413022755027255), 'SynSpansIterations2TxCount15000-10': (0.8659534003246694, 0.06379445126597044), 'SynSpansIterations3TxCount15000-10': (0.8852915502195492, 0.015679366672526087), 'SynSpansIterations4TxCount15000-10': (0.8958942010407537, 0.04781799084874852), 'SynSpansIterations5TxCount15000-10': (0.9241202427869696, 0.06978141712496014), 'SynSpansIterations6TxCount15000-10': (0.8950020537619117, 0.04637683295712088), 'SynSpansIterations10TxCount15000-10': (0.9321221295118929, 0.05800088880532734), 'SynSpansIterations20TxCount15000-10': (0.8702953707284841, 0.05529629928680237), 'SynSpansIterations30TxCount15000-10': (0.9384327556624481, 0.12786818575319708), 'SynSpansIterations40TxCount15000-10': (0.8744524259593786, 0.048482047996782025), 'SynSpansIterations50TxCount15000-10': (0.8055587865074254, 0.097067680511875), 'SynSpansIterations75TxCount15000-10': (0.8092085982070159, 0.07823970528770607), 'SynSpansIterations100TxCount15000-10': (0.9013103646273911, 0.04196112458079615), 'SynSpansIterations150TxCount15000-10': (0.8105378692858322, 0.055120523278858205), 'HeadBasedTraces5': (0.9725920720267356, 0.006710760128807995), 'HeadBasedTraces10': (0.9432894481226655, 0.03222129064353603), 'HeadBasedTraces15': (0.9132021338185361, 0.04361912702312018), 'HeadBasedTraces20': (0.8929583996762064, 0.04885303934698631), 'HeadBasedTraces30': (0.8467279578180451, 0.051919062604145386), 'HeadBasedTraces50': (0.7341667686000304, 0.06455196269145176), 'HeadBasedTraces75': (0.6252291283592422, 0.0785183917740463), 'HeadBasedTraces100': (0.5472607851705404, 0.079251353622372614), 'HeadBasedTraces150': (0.4229719696055785, 0.08509918704579527), 'ErrorBasedTraces': (1.0, 0.0), 'DurationBasedTraces': (0.7242180117269162, 0.11397760219052261)}
        ],
        # raw_sql.bottlenecks_by_time_range(syn_tables, 4, groups=['s1', 's2', 'timeBucket'])
        'Relative duration': [
            {'SynSpansIterations1TxCount15000': (0.3406741396600432, 0.12402288903036456), 'SynSpansIterations2TxCount15000': (0.33828212423565857, 0.11884410017751057), 'SynSpansIterations3TxCount15000': (0.35558973759106044, 0.1354362569541229), 'SynSpansIterations4TxCount15000': (0.3497114458905653, 0.15069136257711271), 'SynSpansIterations5TxCount15000': (0.34384346999192755, 0.12265170331559158), 'SynSpansIterations6TxCount15000': (0.35797600426452765, 0.1318683862520867), 'SynSpansIterations7TxCount15000': (0.3572826552715083, 0.14847907382675432), 'SynSpansIterations10TxCount15000': (0.35321074258803986, 0.1434195848773831), 'SynSpansIterations20TxCount15000': (0.34294171345506824, 0.11984555719743648), 'SynSpansIterations30TxCount15000': (0.36870115272522314, 0.13475893553326052), 'HeadBasedTraces5': (0.24132391577395074, 0.19631638980659097), 'HeadBasedTraces10': (0.36183136845487834, 0.22491777441502225), 'HeadBasedTraces15': (0.40538059549587135, 0.2308927888463764), 'HeadBasedTraces20': (0.4902400267734754, 0.23744521579120345), 'HeadBasedTraces30': (0.5967938203834153, 0.23527846324249163), 'HeadBasedTraces40': (0.6334763545706192, 0.2314471142598331), 'HeadBasedTraces50': (0.6627043622249152, 0.227247133727361), 'HeadBasedTraces75': (0.7339402939078737, 0.21358361724162445), 'HeadBasedTraces100': (0.7599856942866605, 0.20568567830085788), 'HeadBasedTraces150': (0.8106572074085147, 0.18929826634213653), 'ErrorBasedTraces': (0.08231931299496065, 0.11978471232328813), 'DurationBasedTraces': (0.7769444396320847, 0.20580828153247022)},
        ],
        # raw_sql.trigger_correlation(syn_tables, with_sampling=True, hours=4)
        'Trigger discovery': [
            {'SynSpansIterations1TxCount15000': (0.8778213935230619, 0.026255987693993375), 'SynSpansIterations2TxCount15000': (0.8706395348837209, 0.09144032312745176), 'SynSpansIterations3TxCount15000': (0.8735632183908046, 0.08811895335360863), 'SynSpansIterations4TxCount15000': (0.8673344734913266, 0.12852365609972166), 'SynSpansIterations5TxCount15000': (0.8655482616095308, 0.09487663166437783), 'SynSpansIterations6TxCount15000': (0.8732186732186732, 0.1121190054863318), 'SynSpansIterations7TxCount15000': (0.8763878608438194, 0.0641707455734016), 'SynSpansIterations10TxCount15000': (0.8657799274486094, 0.10565199388994154), 'SynSpansIterations20TxCount15000': (0.8589928057553957, 0.10983564320409028), 'SynSpansIterations30TxCount15000': (0.8542422044960116, 0.23387462142609894), 'HeadBasedTraces5': (0.95839721022642, 0.0023825614656467313), 'HeadBasedTraces10': (0.893187232824185, 0.005097814869524016), 'HeadBasedTraces15': (0.8407634428252919, 0.00538002094932892), 'HeadBasedTraces20': (0.7990174334314746, 0.01806115544506096), 'HeadBasedTraces30': (0.7137461345211491, 0.03504516550648455), 'HeadBasedTraces40': (0.6645013018370234, 0.025772508135874204), 'HeadBasedTraces50': (0.6211575459146627, 0.02103583939943244), 'HeadBasedTraces75': (0.5415121805861226, 0.03049675537048234), 'HeadBasedTraces100': (0.4765564047103482, 0.027165024304564456), 'HeadBasedTraces150': (0.38244988587569173, 0.023905410537169873), 'ErrorBasedTraces': (0.9846819246711119, 0.0790805229250867), 'DurationBasedTraces': (0.4336947456213512, 5.551115123125783e-17)}
        ],
        # raw_sql.attributes(syn_tables, attr_name='str_feature_2', with_sampling=True)
        'Attributes': [
            {'SynSpansIterations1TxCount15000': (0.6507267237293415, 0.2705155147866745), 'SynSpansIterations2TxCount15000': (0.7553502534925442, 0.20001993404955689), 'SynSpansIterations3TxCount15000': (0.777223238443669, 0.17808741720411422), 'SynSpansIterations4TxCount15000': (0.7530807357133921, 0.28456909066098767), 'SynSpansIterations5TxCount15000': (0.7781585869522651, 0.3634343437035179), 'SynSpansIterations6TxCount15000': (0.7684625827628606, 0.20635962748146096), 'SynSpansIterations7TxCount15000': (0.7893251212996741, 0.15409916812537366), 'SynSpansIterations10TxCount15000': (0.7942817591135009, 0.2052850063815018), 'SynSpansIterations20TxCount15000': (0.8267436657794049, 0.3975425264387521), 'HeadBasedTraces5': (0.8671386094678368, 0.034036094519931955), 'HeadBasedTraces10': (0.4332724542193268, 0.01634995910451617), 'HeadBasedTraces20': (0.26781065529846093, 0.030305114149187466), 'HeadBasedTraces50': (0.10194609933122131, 0.006613449385203544), 'HeadBasedTraces100': (0.04224716580076064, 0.005274879026833208)}
        ]
    }
    for metric, datas in all_data_for_cost.items():
        if metric != "Monitoring":
            continue
        fig, ax = pyplot.subplots(figsize=(7, 4.3) if e2e else None)
        count_to_iterations: Dict[int, Dict[int, Tuple[float, float]]] = defaultdict(dict)
        for method in datas[0]:
            if "SynSpans" in method:
                iterations, count = re.match(r"SynSpansIterations(\d+)TxCount(\d+)(-10)?", method).groups()[:2]
                avg = numpy.average([float(d[method][0]) for d in datas])
                std = numpy.average([float(d[method][1]) for d in datas])
                count_to_iterations[int(count)][int(iterations)] = avg, std
        for tx_count in ([15000] if e2e else list(count_to_iterations)):
            data = count_to_iterations[tx_count]
            worst_score = 0. if metric != 'Relative duration' else 1.
            ax.errorbar(
                [0.] + [syn_cost(iterations, tx_count) for iterations in data],
                [worst_score] + [avg for avg, std in data.values()],
                [0.] + [std for avg, std in data.values()],
                label=SYS if e2e else tx_count,
            )
            print("cost syn", tx_count, [syn_cost(iterations, tx_count) for iterations in data])
        # put points for the other methods
        if e2e:
            # mark the chosen point with a star
            chosen_x = syn_cost(list(count_to_iterations[15000])[0], 15000)
            chosen_y = list(count_to_iterations[15000].values())[0][0]
            ax.scatter([chosen_x], [chosen_y], marker="*", s=300, color='black', zorder=10)
            ax.hlines(y=chosen_y, xmin=chosen_x, xmax=0.1, linestyle='--', color='grey', alpha=0.5, linewidth=2)
            ax.vlines(x=chosen_x, ymin=0, ymax=chosen_y, linestyle='--', color='grey', alpha=0.5, linewidth=2)

            # add the baselines
            for cost_type, label in (('raw', 'Head sampling'), ('clp', 'Head sampling + CLP'), ('gzip', 'Head sampling + gzip')):
                ratios = []
                x_axis = []
                y_axis = []
                stds = []
                for method in datas[0]:
                    if e2e and "HeadBasedTraces" in method:
                        ratio = int(re.match(r"HeadBasedTraces(\d+)", method).group(1))
                        if (metric != "Attributes" and ratio in (10, 20, 40)) or (metric == 'Relative duration' and ratio == 100):
                            continue
                        ratios.append(ratio)
                        x_axis.append(sampling_cost(ratio, cost_type=cost_type))
                        print("cost", cost_type, ratio, x_axis[-1])
                        y_axis.append(numpy.average([float(d[method][0]) / 1.05 for d in datas]))
                        stds.append(numpy.average([float(d[method][1]) for d in datas]))
                        if (str(ratio), cost_type) in [
                            ("5", "gzip"), ("10", "gzip"), ("15", "gzip"), ("50", "gzip"), ("100", "gzip"), ("150", "gzip"),
                            ("30", "clp"),
                            ("75", "raw"), ("150", "raw"),
                        ]:
                            ax.annotate(f"1:{ratio}", (x_axis[-1] + 0.002, y_axis[-1] - 0.03), fontsize=16)

                ax.errorbar(x_axis, y_axis, stds, marker="o", label=label)
            # for ratio, y in zip(ratios, y_axis):
            #     x = 0.1
            #     ax.hlines(xmin=0, xmax=x, y=y, linestyle='--', color='grey', alpha=0.5, linewidth=1)
            #     ax.annotate(f"1:{ratio}", (x, y - 0.025), fontsize=16)
            # ax.annotate("Sample ratio", (0.09, max(y_axis) + 0.054), fontsize=16)

        ax.set_xlabel("Cost / 1M traces (USD)")
        ax.set_ylabel(f"F1 score")
        ax.set_xlim(0, 0.1)
        ax.set_ylim(-0.05, 1.05)
        legend = ax.legend(ncol=2, fontsize=16, title=None if e2e else "Dataset size")
        pyplot.setp(legend.get_title(), fontsize=20)
        fig.tight_layout()
        if not e2e:
            set_size(ax)
        fig.savefig(f"figures/fidelity_cost{'' if e2e else '_batch'}_{'_'.join(metric.split())}.pdf", format="pdf", bbox_inches='tight')


def chain_length_bar_plot():
    # raw_sql.monitor_chain_length
    result_monitoring = [
        {'SynSpansChainLength2-10': (0.9317973871237906, 0.02927175653321207), 'SynSpansChainLength3-10': (0.9108897225548398, 0.04031558888459952), 'SynSpansChainLength4-10': (0.8186801764162882, 0.08014317355408185), 'SynSpansChainLength5-10': (0.8159396460324604, 0.08105861249794717)},
        {'SynSpansChainLength2-10': (0.8950025111014197, 0.049060061650377275), 'SynSpansChainLength3-10': (0.7556115296000564, 0.09524635833589318), 'SynSpansChainLength4-10': (0.8547085110725522, 0.06496631754580476), 'SynSpansChainLength5-10': (0.7310534043882821, 0.12060852177416435)},
    ]
    result_monitoring = {key.split('-')[0]: (numpy.average([r[key][0] for r in result_monitoring]), numpy.average([r[key][1] for r in result_monitoring])) for key in result_monitoring[0]}
    result_trigger = [
        {'SynSpansChainLength2': (0.8039452870568531, 0.0), 'SynSpansChainLength3': (0.7933660359729036, 0.0), 'SynSpansChainLength4': (0.7971600526612752, 0.0), 'SynSpansChainLength5': (0.7992638601334253, 0.0)},
        {'SynSpansChainLength2': (0.7993529003928819, 0.0), 'SynSpansChainLength3': (0.7956748695003728, 0.0), 'SynSpansChainLength4': (0.8027570076022573, 0.0), 'SynSpansChainLength5': (0.8067655674194463, 0.0)}
    ]
    result_trigger = {key: (numpy.average([r[key][0] for r in result_trigger]), numpy.std([r[key][0] for r in result_trigger])) for key in result_trigger[0]}
    attributes = [
        {'SynSpansChainLength2': (0.9460057019206725, 0.09698527958342743), 'SynSpansChainLength3': (0.9496086806533295, 0.08593315070422651), 'SynSpansChainLength4': (0.8149649744215927, 0.32483000918177823), 'SynSpansChainLength5': (0.7936954617533998, 0.33867702756514423)},
        {'SynSpansChainLength2': (0.9313442344669798, 0.1318324657742064), 'SynSpansChainLength3': (0.8494169747166368, 0.23303062273440797), 'SynSpansChainLength4': (0.9523401798965457, 0.06197360186118707), 'SynSpansChainLength5': (0.8327677773388773, 0.3183595880073739)},
    ]
    attributes = {key: (numpy.average([r[key][0] for r in attributes]), numpy.std([r[key][0] for r in attributes])) for key in attributes[0]}
    duration_ratio = [
        {'SynSpansChainLength2': (0.19620684291736118, 0.1388341676304958), 'SynSpansChainLength3': (0.18355958314268503, 0.161933031855261), 'SynSpansChainLength4': (0.14643324857395884, 0.1453271184252939), 'SynSpansChainLength5': (0.1500369168548929, 0.13224329087401182)},
        {'SynSpansChainLength2': (0.19182483326483754, 0.11813155220686233), 'SynSpansChainLength3': (0.13099805086603952, 0.13470593594001176), 'SynSpansChainLength4': (0.14921541147319264, 0.12040042603716558), 'SynSpansChainLength5': (0.1568112636637867, 0.13190178496451332)},
    ]
    duration_ratio = {key: (numpy.average([r[key][0] for r in duration_ratio]), numpy.std([r[key][0] for r in duration_ratio])) for key in duration_ratio[0]}

    fig, ax = pyplot.subplots()
    ax.errorbar([i for i in range(2, 6)], [result_monitoring[f"SynSpansChainLength{i}"][0] for i in range(2, 6)],
                yerr=[result_monitoring[f"SynSpansChainLength{i}"][1] for i in range(2, 6)], capsize=5, label="Monitoring (F1$\\uparrow$)")
    ax.errorbar([i for i in range(2, 6)], [result_trigger[f"SynSpansChainLength{i}"][0] for i in range(2, 6)],
                yerr=[result_trigger[f"SynSpansChainLength{i}"][1] for i in range(2, 6)], capsize=5, label="Trigger Discovery (F1$\\uparrow$)")
    ax.errorbar([i for i in range(2, 6)], [attributes[f"SynSpansChainLength{i}"][0] for i in range(2, 6)],
                yerr=[attributes[f"SynSpansChainLength{i}"][1] for i in range(2, 6)], capsize=5, label="Attributes (F1$\\uparrow$)")
    ax.errorbar([i for i in range(2, 6)], [duration_ratio[f"SynSpansChainLength{i}"][0] for i in range(2, 6)],
                yerr=[duration_ratio[f"SynSpansChainLength{i}"][1] for i in range(2, 6)], capsize=5, label="Duration Histogram (EMD$\\downarrow$)")

    ax.set_ylabel("Score")
    ax.set_xlabel("Chain length")
    ax.set_xticks([i for i in range(2, 6)])
    ax.legend(loc='center', fontsize=16)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    set_size(ax)
    fig.savefig(r"figures/chain_length_bar.pdf", format="pdf", bbox_inches='tight')


def ablation_bar_plot():
    # raw_sql.simple_ablations
    result_monitoring = [
        {'SynSpansChainLength2-10': (0.8950025111014197, 0.049060061650377275), 'SynSpansAblationNoGCN-10': (0.8721210670312649, 0.047684471680950954), 'SynSpansAblationNoConditioning-10': (0.7146692642998228, 0.12109730540754934), 'SynSpansAblationNotTimeSplit-10': (0.2517406792347308, 0.12798727435743013)}
    ]
    result_monitoring = {key.split('-')[0]: (numpy.average([r[key][0] for r in result_monitoring]), numpy.average([r[key][1] for r in result_monitoring])) for key in result_monitoring[0]}
    result_trigger = [
        {'SynSpansChainLength2': (0.7993529003928819, 0.0082961933319856076), 'SynSpansAblationNoGCN': (0.7991952839898943, 0.0075961933319856076), 'SynSpansAblationNoConditioning': (0.7970262308879226, 0.0081961933319856076), 'SynSpansAblationNotTimeSplit': (0.6052805280528053, 0.0222961933319856076)}
    ]
    result_trigger = {key: (numpy.average([r[key][0] for r in result_trigger]), numpy.average([r[key][1] for r in result_trigger])) for key in result_trigger[0]}
    duration_ratio = [
        {'SynSpansChainLength2': (0.19182483326483754, 0.11813155220686233), 'SynSpansAblationNoGCN': (0.23604373369253328, 0.13090390662923804), 'SynSpansAblationNoConditioning': (0.24243370098012922, 0.1420657180352981), 'SynSpansAblationNotTimeSplit': (0.3793992151568073, 0.33996139310957607)}
    ]
    duration_ratio = {
        key: (numpy.average([r[key][0] for r in duration_ratio]), numpy.average([r[key][1] for r in duration_ratio])) for key in duration_ratio[0]}
    attributes = [
        {'SynSpansChainLength2': (0.9313442344669798, 0.1318324657742064), 'SynSpansAblationNoGCN': (0.8936908651883531, 0.06151908751598532), 'SynSpansAblationNoConditioning': (0.8360674237932322, 0.1581502087385889), 'SynSpansAblationNotTimeSplit': (0.5131598557194855, 0.27535733660389995)}
    ]
    attributes = {key: (numpy.average([r[key][0] for r in attributes]), numpy.average([r[key][1] for r in attributes])) for key in attributes[0]}

    fig, ax = pyplot.subplots()
    width = 1 / len(result_monitoring)
    gap = 1.6
    tick_to_name = {}
    x = 0

    order = ["SynSpansChainLength2", "SynSpansAblationNoGCN", "SynSpansAblationNoConditioning", "SynSpansAblationNotTimeSplit"]
    method_to_name = {
        "SynSpansChainLength2": SYS,
        "SynSpansAblationNoGCN": "1-H Cond.",
        "SynSpansAblationNoConditioning": "No Cond.",
        "SynSpansAblationNotTimeSplit": "Abs. Time"
    }

    for method in order:
        ax.bar(
            [x + i * gap for i in range(4)],
            [result_monitoring[method][0], result_trigger[method][0], attributes[method][0], duration_ratio[method][0]],
            0.1, label=method, alpha=0.9)
        ax.errorbar(
            [x + i * gap for i in range(4)],
            [result_monitoring[method][0], result_trigger[method][0], attributes[method][0], duration_ratio[method][0]],
            yerr=[result_monitoring[method][1], result_trigger[method][1], attributes[method][1], duration_ratio[method][1]],
            fmt='.', capsize=5)
        for i in range(4):
            tick_to_name[x + i * gap] = method_to_name[method]
        x += width

    ax.set_ylabel("Score")
    ax.set_xticks(list(tick_to_name.keys()))
    ax.set_xticklabels(list(tick_to_name.values()), rotation=90, fontsize=16)
    ax.text(0.4, 0.96, "Monitoring\n(F1$\\uparrow$)", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=20)
    ax.text(0.4 + gap, 0.88, "Trigger\nDiscovery\n(F1$\\uparrow$)", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=20)
    ax.text(0.4 + gap * 2, 0.96, "Attributes\n(F1$\\uparrow$)", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=20)
    ax.text(0.4 + gap * 3, 0.88, "Duration\nHistogram\n(EMD$\\downarrow$)", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=20)
    ax.xaxis.grid(False)
    fig.tight_layout()
    set_size(ax)
    fig.savefig(r"figures/ablation_bar.pdf", format="pdf", bbox_inches='tight')


def ctgan_dim_bar_plot():
    # raw_sql.monitor_ctgan_gen_dim
    result_monitoring = [
        {'SynSpansCTGANDim128-10': (0.9317973871237906, 0.02927175653321207), 'SynSpansCTGANDim128_128-10': (0.8202946248491342, 0.07971431128449989), 'SynSpansCTGANDim256_256-10': (0.7051251983675049, 0.13231696112138766), 'SynSpansCTGANDim256-10': (0.8231277017891577, 0.07753778742893926)},
        {'SynSpansCTGANDim128-10': (0.8950025111014197, 0.049060061650377275), 'SynSpansCTGANDim128_128-10': (0.738609754055324, 0.11941430736828322), 'SynSpansCTGANDim256_256-10': (0.9109883932161887, 0.03765925961240356), 'SynSpansCTGANDim256-10': (0.768382480115541, 0.10369101437444002)},
    ]
    result_monitoring = {key.split('-')[0]: (numpy.average([r[key][0] for r in result_monitoring]), numpy.average([r[key][1] for r in result_monitoring])) for key in result_monitoring[0]}
    result_trigger = [
        {'SynSpansCTGANDim128': (0.8039452870568531, 0.0), 'SynSpansCTGANDim128_128': (0.813307130825379, 0.0), 'SynSpansCTGANDim256_256': (0.7880392604428212, 0.0), 'SynSpansCTGANDim256': (0.7978474496958353, 0.0)},
        {'SynSpansCTGANDim128': (0.7993529003928819, 0.0), 'SynSpansCTGANDim128_128': (0.7917149344330662, 0.0), 'SynSpansCTGANDim256_256': (0.7913311553858188, 0.0), 'SynSpansCTGANDim256': (0.8026839426191578, 0.0)},
    ]
    result_trigger = {key: (numpy.average([r[key][0] for r in result_trigger]), numpy.std([r[key][0] for r in result_trigger])) for key in result_trigger[0]}
    duration_ratio = {'SynSpansCTGANDim128': (0.19182483326483754, 0.11813155220686233), 'SynSpansCTGANDim128_128': (0.19447238909769896, 0.17697391841366378), 'SynSpansCTGANDim256_256': (0.18590383509199654, 0.09596222961465935), 'SynSpansCTGANDim256': (0.20952210306761906, 0.13438451253345285)}
    attributes = {'SynSpansCTGANDim128': (0.9313442344669798, 0.1318324657742064), 'SynSpansCTGANDim128_128': (0.8985320814714659, 0.1520695098952055), 'SynSpansCTGANDim256_256': (0.9250103954028002, 0.12263297385886693), 'SynSpansCTGANDim256': (0.9233775719780202, 0.07966339775953306)}

    fig, ax = pyplot.subplots()
    width = 1 / len(result_monitoring)
    gap = 1.6
    tick_to_name = {}
    x = 0

    def label(method: str) -> str:
        res = method[len("SynSpansCTGANDim"):].split('_')
        if len(res) == 1:
            return res[0]
        return f"{res[0]}x{res[1]}"

    order = ["SynSpansCTGANDim128", "SynSpansCTGANDim128_128", "SynSpansCTGANDim256", "SynSpansCTGANDim256_256"]
    for method in order:
        ax.bar(
            [x + i * gap for i in range(4)],
            [result_monitoring[method][0], result_trigger[method][0], attributes[method][0], duration_ratio[method][0]],
            0.1, label=method, alpha=0.9)
        ax.errorbar(
            [x + i * gap for i in range(4)],
            [result_monitoring[method][0], result_trigger[method][0], attributes[method][0], duration_ratio[method][0]],
            yerr=[result_monitoring[method][1], result_trigger[method][1], attributes[method][1], duration_ratio[method][1]],
            fmt='.', capsize=5)
        for i in range(4):
            tick_to_name[x + i * gap] = label(method)
        x += width

    ax.set_ylabel("Score")
    ax.set_xlabel("CTGAN generator dimensions")
    ax.set_xticks(list(tick_to_name.keys()))
    ax.set_xticklabels(list(tick_to_name.values()), rotation=90, fontsize=14)
    ax.text(0.4, 0.96, "Monitoring\n(F1$\\uparrow$)", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=18)
    ax.text(0.4 + gap, 0.88, "Trigger\nDiscovery\n(F1$\\uparrow$)", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=18)
    ax.text(0.4 + gap * 2, 0.96, "Attributes\n(F1$\\uparrow$)", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=18)
    ax.text(0.4 + gap * 3, 0.88, "Duration\nHistogram\n(EMD$\\downarrow$)", ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=18)
    ax.xaxis.grid(False)
    fig.tight_layout()
    set_size(ax)
    fig.savefig(r"figures/ctgan_dim_bar.pdf", format="pdf", bbox_inches='tight')


def rolling_experiment():
    data = {
        "rolled": {
            "trigger_correlation": {'SynRollingSpans0': (0.7465789640467089, 0.04928594973603196), 'SynRollingSpans1': (0.8978033642695907, 0.01916202676937547), 'SynRollingSpans2': (0.8832764984324519, 0.04665418361970433), 'SynRollingSpans3': (0.8884175501246425, 0.028345624976885385), 'RollingSpans0HeadBased5': (0.9064064491657133, 0.04971469254928468), 'RollingSpans0HeadBased10': (0.8188840722562659, 0.06867483784293497), 'RollingSpans0HeadBased20': (0.7112795323634307, 0.09015316297275337), 'RollingSpans0HeadBased50': (0.5050672373127272, 0.10179579141794959), 'RollingSpans0HeadBased100': (0.35973810035100506, 0.08960065651488683), 'RollingSpans1HeadBased5': (0.9337108841683908, 0.036533297087194286), 'RollingSpans1HeadBased10': (0.859163772523979, 0.07402442720243134), 'RollingSpans1HeadBased20': (0.762390086978008, 0.10935763881246865), 'RollingSpans1HeadBased50': (0.5785871613389681, 0.09737663775663885), 'RollingSpans1HeadBased100': (0.43913335723389235, 0.11352638803275772), 'RollingSpans2HeadBased5': (0.9691046228366538, 0.021254534540493867), 'RollingSpans2HeadBased10': (0.9135221487948371, 0.0474499477080809), 'RollingSpans2HeadBased20': (0.8354167316613974, 0.07056989418328473), 'RollingSpans2HeadBased50': (0.663875295612081, 0.08980509590398625), 'RollingSpans2HeadBased100': (0.5221666317101624, 0.09197799066409978), 'RollingSpans3HeadBased5': (0.9333751202855011, 0.040020390715853664), 'RollingSpans3HeadBased10': (0.8603610525709275, 0.07017851543536606), 'RollingSpans3HeadBased20': (0.746917219464604, 0.08411253863761695), 'RollingSpans3HeadBased50': (0.5741225549859925, 0.10882995319192935), 'RollingSpans3HeadBased100': (0.42171582806976027, 0.09751193287184833)},
            "monitor_errors": {'SynRollingSpans0-10': (0.7536391953296108, 0.08107086898839955), 'SynRollingSpans1-10': (0.8709879764337052, 0.042132972075484824), 'SynRollingSpans2-10': (0.8752944356401491, 0.05671046836061877), 'SynRollingSpans3-10': (0.8372820811492914, 0.03320277067945805), 'RollingSpans0HeadBased5-10': (0.7189048163890636, 0.11982619688394404), 'RollingSpans0HeadBased10-10': (0.47850935768684233, 0.15923108334486144), 'RollingSpans0HeadBased20-10': (0.2705764716135176, 0.152815709962471), 'RollingSpans0HeadBased50-10': (0.06652064206983987, 0.07326591862980833), 'RollingSpans0HeadBased100-10': (0.02358074321011864, 0.03572056883307799), 'RollingSpans1HeadBased5-10': (0.8492351097678011, 0.0546553239026705), 'RollingSpans1HeadBased10-10': (0.6098709983252846, 0.15681020252774155), 'RollingSpans1HeadBased20-10': (0.33889968739346554, 0.1743920915146213), 'RollingSpans1HeadBased50-10': (0.08922898492173306, 0.12006993951452723), 'RollingSpans1HeadBased100-10': (0.03260869565217391, 0.05647991763811556), 'RollingSpans2HeadBased5-10': (0.9055032183288945, 0.027176816380577118), 'RollingSpans2HeadBased10-10': (0.7318103374152654, 0.0992382006930203), 'RollingSpans2HeadBased20-10': (0.5063723790621274, 0.14744080030444806), 'RollingSpans2HeadBased50-10': (0.21946735529399738, 0.12773054741703385), 'RollingSpans2HeadBased100-10': (0.07183642233912112, 0.08044615618265337), 'RollingSpans3HeadBased5-10': (0.608440427383763, 0.14551952231031698), 'RollingSpans3HeadBased10-10': (0.36554442242549945, 0.1410926438329705), 'RollingSpans3HeadBased20-10': (0.1323282416515606, 0.1107837268504137), 'RollingSpans3HeadBased50-10': (0.04155667701863354, 0.057813610269066104), 'RollingSpans3HeadBased100-10': (0.011428571428571429, 0.019794866372215742)},
            "bottlenecks_by_time_range": {'SynRollingSpans0': (0.22595141909614572, 0.21070796102526626), 'SynRollingSpans1': (0.20090835530421666, 0.14349379263152467), 'SynRollingSpans2': (0.20051943081701368, 0.14308524556189847), 'SynRollingSpans3': (0.2068504074649906, 0.1599745587698087), 'RollingSpans0HeadBased5': (0.11138399631771838, 0.2681402229700263), 'RollingSpans0HeadBased10': (0.21165034383684936, 0.36451635234051727), 'RollingSpans0HeadBased20': (0.37364483526894227, 0.45426556152578973), 'RollingSpans0HeadBased50': (0.5175803662566225, 0.47610794009486396), 'RollingSpans0HeadBased100': (0.6305607824266795, 0.4619036797446127), 'RollingSpans1HeadBased5': (0.07405075292194656, 0.21147038100428645), 'RollingSpans1HeadBased10': (0.09596585874815022, 0.2402637033135747), 'RollingSpans1HeadBased20': (0.26815373553168453, 0.40384861150816903), 'RollingSpans1HeadBased50': (0.36310887154716687, 0.4481489485017139), 'RollingSpans1HeadBased100': (0.5012596020969133, 0.4726154674304158), 'RollingSpans2HeadBased5': (0.03855940440386247, 0.13919122543363957), 'RollingSpans2HeadBased10': (0.10692023168622881, 0.2693100590701754), 'RollingSpans2HeadBased20': (0.15887485205982038, 0.31765250454229194), 'RollingSpans2HeadBased50': (0.3406863370834439, 0.4347866805666383), 'RollingSpans2HeadBased100': (0.4055711667951714, 0.4580540832841976), 'RollingSpans3HeadBased5': (0.06465700253619083, 0.19733805769174403), 'RollingSpans3HeadBased10': (0.151284580024408, 0.31625353675707263), 'RollingSpans3HeadBased20': (0.25125087688357006, 0.39229621165480205), 'RollingSpans3HeadBased50': (0.3691039819269766, 0.4387318804806275), 'RollingSpans3HeadBased100': (0.5185962588675128, 0.46383464269190755)},
            "attributes": {'SynRollingSpans0': (0.9159554471221634, 0.1173330303645998), 'SynRollingSpans1': (0.8842470222445662, 0.1874125271909871), 'SynRollingSpans2': (0.8662653927537925, 0.03089924929370579), 'SynRollingSpans3': (0.8909965162205241, 0.048139602899751964), 'RollingSpans0HeadBased5': (0.8671386094678368, 0.034036094519931955), 'RollingSpans0HeadBased10': (0.4332724542193268, 0.01634995910451617), 'RollingSpans0HeadBased20': (0.26781065529846093, 0.030305114149187466), 'RollingSpans0HeadBased50': (0.10194609933122131, 0.006613449385203544), 'RollingSpans0HeadBased100': (0.04224716580076064, 0.005274879026833208), 'RollingSpans1HeadBased5': (0.8397476266481446, 0.02307220165205552), 'RollingSpans1HeadBased10': (0.48862169131713423, 0.018626701330306295), 'RollingSpans1HeadBased20': (0.24432654675416418, 0.017780296536868707), 'RollingSpans1HeadBased50': (0.10053655334356822, 0.006687158886595081), 'RollingSpans1HeadBased100': (0.04593558770623743, 0.0026994790547064376), 'RollingSpans2HeadBased5': (0.8059353669562435, 0.021841077272220615), 'RollingSpans2HeadBased10': (0.4761996328993995, 0.01488041225809491), 'RollingSpans2HeadBased20': (0.2396220618820057, 0.01036564302380223), 'RollingSpans2HeadBased50': (0.08983623240035307, 0.005893397013326533), 'RollingSpans2HeadBased100': (0.0470926887339717, 0.005862571852925171), 'RollingSpans3HeadBased5': (0.8516630566207419, 0.010970200887860623), 'RollingSpans3HeadBased10': (0.4449765896065905, 0.010393344126398377), 'RollingSpans3HeadBased20': (0.22112344283907695, 0.013762376377701089), 'RollingSpans3HeadBased50': (0.10087555616123249, 0.006117439530217669), 'RollingSpans3HeadBased100': (0.04758283339246808, 0.005300848479322503)},
        }
    }
    metrics = {
        "monitor_errors": ("Monitoring", "F1($\\uparrow$)"),
        "trigger_correlation": ("Trigger Discovery", "F1($\\uparrow$)"),
        "attributes": ("Categorical Attributes", "F1($\\uparrow$)"),
        "bottlenecks_by_time_range": ("Duration Histogram", "EMD($\\downarrow$)"),
    }
    comparisons = {
        'SynRollingSpans0': SYS,
        'RollingSpans0HeadBased5': '1:5',
        'RollingSpans0HeadBased20': '1:20',
        'RollingSpans0HeadBased50': '1:50',
        'RollingSpans0HeadBased100': '1:100',
    }

    data = {k1: {k2: {k3.split('-')[0]: v3 for k3, v3 in v2.items()} for k2, v2 in v1.items()} for k1, v1 in data.items()}

    fig, ax = pyplot.subplots(2, 2, sharex=True, figsize=(10, 4))
    texts = []
    for ax_index, (metric, (name, y_label)) in enumerate(metrics.items()):
        cax = ax[ax_index // 2][ax_index % 2]
        for cmp, cmp_name in comparisons.items():
            cax.errorbar(
                list(range(4)),
                [data["rolled"][metric][cmp.replace('Spans0', f'Spans{j}')][0] for j in range(4)],
                [data["rolled"][metric][cmp.replace('Spans0', f'Spans{j}')][1] for j in range(4)],
                label=cmp_name if ax_index == 0 else None,
                capsize=3,
                alpha=0.7,
            )
            [cax.vlines(x=j + 0.5, ymin=-1, ymax=2, linestyle='--', color='grey', alpha=0.5, linewidth=1, clip_on=True) for j in range(3)]
            cax.xaxis.grid(False)

        cax.set_ylim(-0.05, 1.1)
        cax.set_xlim(-0.5, 3.5)
        cax.set_yticks([0, 1])
        cax.set_xticks(list(range(4)))
        cax.set_xticklabels(["Period 1:\nNo\nchange", "Period 2:\n2X spike\nin errors",
                             "Period 3:\n2X spike\nin #\nof traces", "Period 4:\n2X less\nerrors"],
                            rotation=0, fontsize=14)
        cax.arrow(-0.5, -0.005, 3.98, 0., fc='k', ec='k', head_width=1/20, head_length=1/5, overhang=0.6, length_includes_head=True, clip_on=False)
        cax.annotate("Time", xy=(0.95, 0.08), xycoords='axes fraction', ha='center', fontsize=12)
        texts.append(cax.annotate(name, xy=(0.5, 1.01), xycoords='axes fraction', ha='center', fontsize=15))
        texts.append(cax.annotate(y_label, xy=(-0.02, 0.45), xycoords='axes fraction', ha='center', fontsize=15))

    [ax[1][i].vlines(x=j + 0.5, ymin=-0.5, ymax=0, linestyle='--', color='grey', alpha=0.5, linewidth=1, clip_on=False) for j in range(3) for i in range(2)]

    legend = fig.legend(fontsize=12, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(r"figures/rolling_experiment.pdf", bbox_extra_artists=[legend]+texts, format="pdf", bbox_inches='tight')


def exp_lambda_graph_by_query():
    raw_data = {
        0.001 : {
"Find traces of a service that having an error" : {
"DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0011858712708590039,
"DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0012599020597356427,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0008047361581811664,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.004008134107042174,
},

"Find traces that have a particular attribute" : {
"DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.026930670567044127,
"DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.03271909926366265,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.02481140323101613,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.03310859659302862,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.015249595165652954,
},

"Discover architecture of the whole system" : {
"DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.00028083818066663383,
"DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.00028163708796311314,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0002807253179897148,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.00028122549614065684,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0002818972094494428,
},

"Find bottlenecks" : {
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.013386061008974342,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0076905982736220554,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.012352830306100222,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.008464302033896375,
"DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.00649321356489903,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.00739800636224236,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.011436605044634228,
"DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.010729428570233778,
},

"RED metrics - rate" : {
"DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.00014242569125702903,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.00014211590358412558,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.00014243991892466388,
"DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.00014246158046248846,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.00014169332219921178,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.00014250594283678387,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0001425949199039606,
"DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.00014297498358949566,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.00014330547505852156,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.00014228831389300747,
"DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.00014243386085400922,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.00014196431287907582,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.00014296701218099008,
},

"RED metrics - error" : {
"DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.006465274080417337,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.00401146157977044,
"DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0036051893377020963,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0069748155822993195,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.003300008558563111,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.002701837090402218,
"DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0033632816064078632,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0037229653979059083,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.004228343020087873,
},

"RED metrics - duration" : {
"DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0016126559879899428,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0023355647430803174,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.001661388061824301,
"DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.001657074346208593,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.002405542655358491,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.00175596001347933,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0015200815916595397,
"DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0016168215810193963,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0017360656803111763,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0018910187223659199,
"DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0019211233836726788,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.002193982640683963,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0019859456127393144,
},

"Frequency of an attribute" : {
"DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.017786701880886803,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.021968473630057498,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.020458346619173486,
"DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.020786691421161248,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.023677480061503265,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.022690379848432547,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.015654457562005723,
"DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.01664421819361051,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.02246331606858321,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.021733224918870003,
"DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.022020685473775132,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.021753181360130512,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.00832878450186581,
},

"Max value of an attribute for every 5 minute window" : {
"DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.00020683431141293226,
"DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.00028640206733914587,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.00033734874817549867,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0003812511775169878,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0003005577308755614,
},

"Frequency of an attribute after filtering by another attribute" : {
"DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.13399833738305847,
"DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.1195355019765158,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.08682150451940514,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.13095707952868443,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.06918115121235473,
},

},


0.002 : {
"Find traces of a service that having an error" : {
"DeathStarExpLambdasocialNetwork_packet_loss_02_0002_0": 0.0027856469024895043,
},

"Find traces that have a particular attribute" : {
"DeathStarExpLambdasocialNetwork_packet_loss_02_0002_0": 0.02917173263692989,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0002_0": 0.03335519215780923,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0002_0": 0.03512485742927835,
"DeathStarExpLambdasocialNetwork_crush_01_0002_0": 0.03463218897437442,
"DeathStarExpLambdasocialNetwork_crush_02_0002_0": 0.035696161796916585,
},

"Discover architecture of the whole system" : {
"DeathStarExpLambdasocialNetwork_packet_loss_02_0002_0": 0.00028158295282470103,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0002_0": 0.0002811840785743702,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0002_0": 0.0002826954257010736,
"DeathStarExpLambdasocialNetwork_crush_01_0002_0": 0.00028003116801805946,
"DeathStarExpLambdasocialNetwork_crush_02_0002_0": 0.0002817414481178276,
},

"Find bottlenecks" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_0002_0": 0.008506148537275595,
"DeathStarExpLambdasocialNetwork_latency_02_0002_0": 0.007695078899576557,
"DeathStarExpLambdasocialNetwork_latency_01_0002_0": 0.006997988577172389,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0002_0": 0.007578157487524397,
"DeathStarExpLambdasocialNetwork_latency_03_0002_0": 0.008966169159477587,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0002_0": 0.009959384522590532,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0002_0": 0.009802578042036783,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0002_0": 0.008309849800490397,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0002_0": 0.011433742147179219,
},

"RED metrics - rate" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_0002_0": 0.0001422354475246241,
"DeathStarExpLambdasocialNetwork_latency_02_0002_0": 0.00014202645398960333,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0002_0": 0.00014261629326290347,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0002_0": 0.0001420343883185289,
"DeathStarExpLambdasocialNetwork_latency_01_0002_0": 0.00014319569714923094,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0002_0": 0.0001428868805442396,
"DeathStarExpLambdasocialNetwork_latency_03_0002_0": 0.00014209631094694224,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0002_0": 0.00014296304924546333,
"DeathStarExpLambdasocialNetwork_crush_01_0002_0": 0.0001428512336996177,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0002_0": 0.00014288522818903848,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0002_0": 0.00014214907978859579,
"DeathStarExpLambdasocialNetwork_crush_02_0002_0": 0.00014206878268267382,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0002_0": 0.00014268415045403938,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0002_0": 0.0001429585961620279,
},

"RED metrics - error" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_0002_0": 0.0036813908932661057,
"DeathStarExpLambdasocialNetwork_latency_02_0002_0": 0.003460311728952261,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0002_0": 0.003939173410938308,
"DeathStarExpLambdasocialNetwork_latency_01_0002_0": 0.00764679698372817,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0002_0": 0.00500240272886974,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0002_0": 0.0050151639603209465,
},

"RED metrics - duration" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_0002_0": 0.0016798866139265693,
"DeathStarExpLambdasocialNetwork_latency_02_0002_0": 0.0017733724263058962,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0002_0": 0.0015636729005335786,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0002_0": 0.0019009599676151564,
"DeathStarExpLambdasocialNetwork_latency_01_0002_0": 0.0016562957029283999,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0002_0": 0.001566016158988083,
"DeathStarExpLambdasocialNetwork_latency_03_0002_0": 0.0016027460126549038,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0002_0": 0.0030255968130407907,
"DeathStarExpLambdasocialNetwork_crush_01_0002_0": 0.0020106894710862273,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0002_0": 0.002059122704664194,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0002_0": 0.0017171192425942568,
"DeathStarExpLambdasocialNetwork_crush_02_0002_0": 0.0019513879208698114,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0002_0": 0.0016716174811647673,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0002_0": 0.00172596273564672,
},

"Frequency of an attribute" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_0002_0": 0.017410974829342306,
"DeathStarExpLambdasocialNetwork_latency_02_0002_0": 0.017963185177334095,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0002_0": 0.0210058112636033,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0002_0": 0.02176907060348133,
"DeathStarExpLambdasocialNetwork_latency_01_0002_0": 0.02490332828449381,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0002_0": 0.018919750511195526,
"DeathStarExpLambdasocialNetwork_latency_03_0002_0": 0.02382773272834671,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0002_0": 0.022690548345044456,
"DeathStarExpLambdasocialNetwork_crush_01_0002_0": 0.02287097789211298,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0002_0": 0.022721249872311076,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0002_0": 0.017100985944195637,
"DeathStarExpLambdasocialNetwork_crush_02_0002_0": 0.02299539072773231,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0002_0": 0.019351996890092496,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0002_0": 0.017216955490969094,
},

"Max value of an attribute for every 5 minute window" : {
"DeathStarExpLambdasocialNetwork_packet_loss_02_0002_0": 0.0002411152618309424,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0002_0": 0.0003679242674821649,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0002_0": 0.00028289234792126155,
"DeathStarExpLambdasocialNetwork_crush_01_0002_0": 0.00022644970948255746,
"DeathStarExpLambdasocialNetwork_crush_02_0002_0": 0.00029920823324921354,
},

"Frequency of an attribute after filtering by another attribute" : {
"DeathStarExpLambdasocialNetwork_packet_loss_02_0002_0": 0.12807741745741785,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0002_0": 0.1586408276960375,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0002_0": 0.16958408492060273,
"DeathStarExpLambdasocialNetwork_crush_01_0002_0": 0.1624689836490694,
"DeathStarExpLambdasocialNetwork_crush_02_0002_0": 0.15258171530380613,
},

},


0.005 : {
"Find traces of a service that having an error" : {
"DeathStarExpLambdasocialNetwork_packet_loss_03_0005_0": 0.0026215493918788843,
"DeathStarExpLambdasocialNetwork_crush_02_0005_0": 0.001081968230657981,
},

"Find traces that have a particular attribute" : {
"DeathStarExpLambdasocialNetwork_packet_loss_02_0005_0": 0.029400173462393726,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0005_0": 0.029779335938168536,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0005_0": 0.033562049869745945,
"DeathStarExpLambdasocialNetwork_crush_01_0005_0": 0.02881246755500605,
"DeathStarExpLambdasocialNetwork_crush_02_0005_0": 0.02842878216521971,
},

"Discover architecture of the whole system" : {
"DeathStarExpLambdasocialNetwork_packet_loss_02_0005_0": 0.0002788937923461862,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0005_0": 0.0002795013973612369,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0005_0": 0.00028226634564320274,
"DeathStarExpLambdasocialNetwork_crush_01_0005_0": 0.00027752363678962115,
"DeathStarExpLambdasocialNetwork_crush_02_0005_0": 0.0002818779429304897,
},

"Find bottlenecks" : {
"DeathStarExpLambdasocialNetwork_latency_02_0005_0": 0.011900180939940682,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0005_0": 0.007663939772367471,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0005_0": 0.00846707022019553,
"DeathStarExpLambdasocialNetwork_latency_01_0005_0": 0.007529227866683269,
"DeathStarExpLambdasocialNetwork_latency_03_0005_0": 0.0065657293361483515,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0005_0": 0.008650435792364,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0005_0": 0.007922454716835836,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0005_0": 0.006450273550393247,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0005_0": 0.0078091739086360405,
},

"RED metrics - rate" : {
"DeathStarExpLambdasocialNetwork_latency_02_0005_0": 0.00014263883669115603,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0005_0": 0.00014258399756072576,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0005_0": 0.00014196679464102504,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0005_0": 0.0001423534350245447,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0005_0": 0.0001427243438176854,
"DeathStarExpLambdasocialNetwork_latency_01_0005_0": 0.0001428562229137887,
"DeathStarExpLambdasocialNetwork_latency_03_0005_0": 0.00014161044352241302,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0005_0": 0.0001415729774904324,
"DeathStarExpLambdasocialNetwork_crush_01_0005_0": 0.00014254142922479784,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0005_0": 0.00014307689551398625,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0005_0": 0.00014229450147448672,
"DeathStarExpLambdasocialNetwork_crush_02_0005_0": 0.00014218743909099712,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0005_0": 0.00014267791626112808,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0005_0": 0.00014309947429790246,
},

"RED metrics - error" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_0005_0": 0.0030340931086121,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0005_0": 0.0037975103681627377,
"DeathStarExpLambdasocialNetwork_latency_01_0005_0": 0.0036343751224978856,
"DeathStarExpLambdasocialNetwork_latency_03_0005_0": 0.0034375397995210955,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0005_0": 0.006210621508936424,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0005_0": 0.003641020740656414,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0005_0": 0.0038338249909635755,
"DeathStarExpLambdasocialNetwork_crush_02_0005_0": 0.0028675987898635047,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0005_0": 0.0034602467404701706,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0005_0": 0.003458137565857936,
},

"RED metrics - duration" : {
"DeathStarExpLambdasocialNetwork_latency_02_0005_0": 0.002014854363043366,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0005_0": 0.0016923981320514567,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0005_0": 0.0017784377811648577,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0005_0": 0.0023706697208856512,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0005_0": 0.0018714964058343625,
"DeathStarExpLambdasocialNetwork_latency_01_0005_0": 0.0017055679918581462,
"DeathStarExpLambdasocialNetwork_latency_03_0005_0": 0.0015726317160247812,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0005_0": 0.0018141669482946044,
"DeathStarExpLambdasocialNetwork_crush_01_0005_0": 0.001860345973952019,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0005_0": 0.0015466284015209326,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0005_0": 0.001735794953818191,
"DeathStarExpLambdasocialNetwork_crush_02_0005_0": 0.0014279418577362577,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0005_0": 0.0016934519939458012,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0005_0": 0.0016135797303666554,
},

"Frequency of an attribute" : {
"DeathStarExpLambdasocialNetwork_latency_02_0005_0": 0.02006237233552323,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0005_0": 0.018841279562509666,
"DeathStarExpLambdasocialNetwork_packet_loss_02_0005_0": 0.018777647345443724,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0005_0": 0.01895969299178363,
"DeathStarExpLambdasocialNetwork_memory_stress_02_0005_0": 0.01829284552668539,
"DeathStarExpLambdasocialNetwork_latency_01_0005_0": 0.019003546676828555,
"DeathStarExpLambdasocialNetwork_latency_03_0005_0": 0.017077127658056492,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0005_0": 0.02168899645669302,
"DeathStarExpLambdasocialNetwork_crush_01_0005_0": 0.018236303038769854,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0005_0": 0.02046343448506387,
"DeathStarExpLambdasocialNetwork_cpu_load_02_0005_0": 0.022399409895400763,
"DeathStarExpLambdasocialNetwork_crush_02_0005_0": 0.01622896677080575,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_0005_0": 0.022238403718428668,
"DeathStarExpLambdasocialNetwork_cpu_load_01_0005_0": 0.02101471262742571,
},

"Max value of an attribute for every 5 minute window" : {
"DeathStarExpLambdasocialNetwork_packet_loss_02_0005_0": 0.0002618238507532573,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0005_0": 0.00023293192278833885,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0005_0": 0.00022912752158807775,
"DeathStarExpLambdasocialNetwork_crush_01_0005_0": 0.00024284867991501265,
"DeathStarExpLambdasocialNetwork_crush_02_0005_0": 0.0003452977972529095,
},

"Frequency of an attribute after filtering by another attribute" : {
"DeathStarExpLambdasocialNetwork_packet_loss_02_0005_0": 0.14224212965060076,
"DeathStarExpLambdasocialNetwork_packet_loss_01_0005_0": 0.15449449822131392,
"DeathStarExpLambdasocialNetwork_packet_loss_03_0005_0": 0.13969451510926587,
"DeathStarExpLambdasocialNetwork_crush_01_0005_0": 0.14508177187187807,
"DeathStarExpLambdasocialNetwork_crush_02_0005_0": 0.11016314504915221,
},

},


0.01 : {
"Find traces of a service that having an error" : {
"DeathStarExpLambdasocialNetwork_packet_loss_03_001_0": 0.0008015431775271952,
},

"Find traces that have a particular attribute" : {
"DeathStarExpLambdasocialNetwork_crush_01_001_0": 0.031270283726347985,
"DeathStarExpLambdasocialNetwork_packet_loss_02_001_0": 0.021959624865231602,
"DeathStarExpLambdasocialNetwork_packet_loss_03_001_0": 0.03227830590911479,
"DeathStarExpLambdasocialNetwork_packet_loss_01_001_0": 0.031061587287191227,
},

"Discover architecture of the whole system" : {
"DeathStarExpLambdasocialNetwork_crush_01_001_0": 0.0002748476861422545,
"DeathStarExpLambdasocialNetwork_packet_loss_02_001_0": 0.00027767014252096545,
"DeathStarExpLambdasocialNetwork_packet_loss_03_001_0": 0.0002812877243229051,
"DeathStarExpLambdasocialNetwork_packet_loss_01_001_0": 0.00027756745356412565,
},

"Find bottlenecks" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_001_0": 0.011731134330562585,
"DeathStarExpLambdasocialNetwork_latency_03_001_0": 0.007713179854865194,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_001_0": 0.006925836896881959,
"DeathStarExpLambdasocialNetwork_cpu_load_01_001_0": 0.011111473478073938,
"DeathStarExpLambdasocialNetwork_latency_02_001_0": 0.007850839249155042,
"DeathStarExpLambdasocialNetwork_latency_01_001_0": 0.0076577661501799395,
"DeathStarExpLambdasocialNetwork_cpu_load_02_001_0": 0.007744950220102614,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_001_0": 0.007991626323120693,
"DeathStarExpLambdasocialNetwork_memory_stress_02_001_0": 0.007772099399210654,
},

"RED metrics - rate" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_001_0": 0.00014112259983920756,
"DeathStarExpLambdasocialNetwork_crush_01_001_0": 0.00014104209577046585,
"DeathStarExpLambdasocialNetwork_latency_03_001_0": 0.0001427102501784144,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_001_0": 0.00014234756594568972,
"DeathStarExpLambdasocialNetwork_cpu_load_01_001_0": 0.00014315040760678907,
"DeathStarExpLambdasocialNetwork_packet_loss_02_001_0": 0.00014241096436250995,
"DeathStarExpLambdasocialNetwork_packet_loss_03_001_0": 0.00014136022015107168,
"DeathStarExpLambdasocialNetwork_latency_02_001_0": 0.00014192066754577554,
"DeathStarExpLambdasocialNetwork_latency_01_001_0": 0.00014228275824794295,
"DeathStarExpLambdasocialNetwork_cpu_load_02_001_0": 0.00014198448873583306,
"DeathStarExpLambdasocialNetwork_packet_loss_01_001_0": 0.00014242748420413828,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_001_0": 0.0001421579872970937,
"DeathStarExpLambdasocialNetwork_memory_stress_02_001_0": 0.00014197873996812376,
},

"RED metrics - error" : {
"DeathStarExpLambdasocialNetwork_latency_03_001_0": 0.0034306761876271535,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_001_0": 0.0033260452364721445,
"DeathStarExpLambdasocialNetwork_packet_loss_03_001_0": 0.0030734409476733757,
"DeathStarExpLambdasocialNetwork_latency_02_001_0": 0.0034902396837163847,
"DeathStarExpLambdasocialNetwork_latency_01_001_0": 0.0050657660860291415,
"DeathStarExpLambdasocialNetwork_cpu_load_02_001_0": 0.005340507537361055,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_001_0": 0.0054784745196055015,
"DeathStarExpLambdasocialNetwork_memory_stress_02_001_0": 0.003083668716314291,
},

"RED metrics - duration" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_001_0": 0.001697359868658399,
"DeathStarExpLambdasocialNetwork_crush_01_001_0": 0.0017359332430388155,
"DeathStarExpLambdasocialNetwork_latency_03_001_0": 0.0015068170201246575,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_001_0": 0.0015335572449784816,
"DeathStarExpLambdasocialNetwork_cpu_load_01_001_0": 0.0021398271378006367,
"DeathStarExpLambdasocialNetwork_packet_loss_02_001_0": 0.0017885382584916715,
"DeathStarExpLambdasocialNetwork_packet_loss_03_001_0": 0.0014993706865701342,
"DeathStarExpLambdasocialNetwork_latency_02_001_0": 0.0014584494885618372,
"DeathStarExpLambdasocialNetwork_latency_01_001_0": 0.0020754174863386234,
"DeathStarExpLambdasocialNetwork_cpu_load_02_001_0": 0.0019227962757931872,
"DeathStarExpLambdasocialNetwork_packet_loss_01_001_0": 0.0021658761017180404,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_001_0": 0.001903283471616349,
"DeathStarExpLambdasocialNetwork_memory_stress_02_001_0": 0.0014761933081697376,
},

"Frequency of an attribute" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_001_0": 0.022489980039539104,
"DeathStarExpLambdasocialNetwork_crush_01_001_0": 0.020772654769518322,
"DeathStarExpLambdasocialNetwork_latency_03_001_0": 0.020037698712716018,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_001_0": 0.016396940499478558,
"DeathStarExpLambdasocialNetwork_cpu_load_01_001_0": 0.019668886571122098,
"DeathStarExpLambdasocialNetwork_packet_loss_02_001_0": 0.013310170348435766,
"DeathStarExpLambdasocialNetwork_packet_loss_03_001_0": 0.022307089365678393,
"DeathStarExpLambdasocialNetwork_latency_02_001_0": 0.02365094050879385,
"DeathStarExpLambdasocialNetwork_latency_01_001_0": 0.02052194886366336,
"DeathStarExpLambdasocialNetwork_cpu_load_02_001_0": 0.022805115793010502,
"DeathStarExpLambdasocialNetwork_packet_loss_01_001_0": 0.01962104579632375,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_001_0": 0.02058834572197454,
"DeathStarExpLambdasocialNetwork_memory_stress_02_001_0": 0.02448351523478753,
},

"Max value of an attribute for every 5 minute window" : {
"DeathStarExpLambdasocialNetwork_crush_01_001_0": 0.00024391432857910337,
"DeathStarExpLambdasocialNetwork_packet_loss_02_001_0": 0.00029787915576565897,
"DeathStarExpLambdasocialNetwork_packet_loss_03_001_0": 0.0003469130857109874,
"DeathStarExpLambdasocialNetwork_packet_loss_01_001_0": 0.0002686426554470778,
},

"Frequency of an attribute after filtering by another attribute" : {
"DeathStarExpLambdasocialNetwork_crush_01_001_0": 0.1622062261794729,
"DeathStarExpLambdasocialNetwork_packet_loss_02_001_0": 0.10238086576808708,
"DeathStarExpLambdasocialNetwork_packet_loss_03_001_0": 0.1377494348741316,
"DeathStarExpLambdasocialNetwork_packet_loss_01_001_0": 0.16257697614434866,
},

},


0.02 : {
"Find traces of a service that having an error" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_002_0": 0.005922538068418318,
"DeathStarExpLambdasocialNetwork_crush_01_002_0": 0.0007522571863885976,
"DeathStarExpLambdasocialNetwork_packet_loss_02_002_0": 0.0018899636226562318,
"DeathStarExpLambdasocialNetwork_packet_loss_03_002_0": 0.004613323239265953,
},

"Find traces that have a particular attribute" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_002_0": 0.030914124887010227,
"DeathStarExpLambdasocialNetwork_crush_02_002_0": 0.04214443490519643,
"DeathStarExpLambdasocialNetwork_crush_01_002_0": 0.03634083994405776,
"DeathStarExpLambdasocialNetwork_packet_loss_02_002_0": 0.03905195016745224,
"DeathStarExpLambdasocialNetwork_packet_loss_03_002_0": 0.028755355806446725,
},

"Discover architecture of the whole system" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_002_0": 0.0002740077109476442,
"DeathStarExpLambdasocialNetwork_crush_02_002_0": 0.0002845345760363994,
"DeathStarExpLambdasocialNetwork_crush_01_002_0": 0.0002721022182489817,
"DeathStarExpLambdasocialNetwork_packet_loss_02_002_0": 0.0002783058433115639,
"DeathStarExpLambdasocialNetwork_packet_loss_03_002_0": 0.0002835943578427081,
},

"Find bottlenecks" : {
"DeathStarExpLambdasocialNetwork_latency_01_002_0": 0.013994199616192762,
"DeathStarExpLambdasocialNetwork_cpu_load_02_002_0": 0.007981201011665318,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_002_0": 0.006167025779734103,
"DeathStarExpLambdasocialNetwork_memory_stress_02_002_0": 0.008794323164249841,
"DeathStarExpLambdasocialNetwork_memory_stress_01_002_0": 0.009136784264135358,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_002_0": 0.007731902248916324,
"DeathStarExpLambdasocialNetwork_latency_03_002_0": 0.011589341995080651,
"DeathStarExpLambdasocialNetwork_cpu_load_01_002_0": 0.008336181302422316,
"DeathStarExpLambdasocialNetwork_latency_02_002_0": 0.01051970756886426,
},

"RED metrics - rate" : {
"DeathStarExpLambdasocialNetwork_latency_01_002_0": 0.00013983873832398204,
"DeathStarExpLambdasocialNetwork_packet_loss_01_002_0": 0.00014119458628442195,
"DeathStarExpLambdasocialNetwork_cpu_load_02_002_0": 0.00014024404185040198,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_002_0": 0.00013901310487529728,
"DeathStarExpLambdasocialNetwork_crush_02_002_0": 0.00014224554026742142,
"DeathStarExpLambdasocialNetwork_memory_stress_02_002_0": 0.00014021553584214814,
"DeathStarExpLambdasocialNetwork_crush_01_002_0": 0.00014184625962704805,
"DeathStarExpLambdasocialNetwork_memory_stress_01_002_0": 0.00013752144887755222,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_002_0": 0.00013685996421233449,
"DeathStarExpLambdasocialNetwork_latency_03_002_0": 0.0001384996650976472,
"DeathStarExpLambdasocialNetwork_packet_loss_02_002_0": 0.00014188525447581237,
"DeathStarExpLambdasocialNetwork_cpu_load_01_002_0": 0.0001380432667137152,
"DeathStarExpLambdasocialNetwork_packet_loss_03_002_0": 0.00013886498901797414,
"DeathStarExpLambdasocialNetwork_latency_02_002_0": 0.00014046823480828966,
},

"RED metrics - error" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_002_0": 0.003258603727632191,
"DeathStarExpLambdasocialNetwork_cpu_load_02_002_0": 0.004370074172015942,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_002_0": 0.004456220176249544,
"DeathStarExpLambdasocialNetwork_crush_01_002_0": 0.007435745785559777,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_002_0": 0.0034828159330849754,
"DeathStarExpLambdasocialNetwork_packet_loss_02_002_0": 0.005184279740244537,
"DeathStarExpLambdasocialNetwork_cpu_load_01_002_0": 0.011805015608081625,
"DeathStarExpLambdasocialNetwork_packet_loss_03_002_0": 0.003687422810858166,
},

"RED metrics - duration" : {
"DeathStarExpLambdasocialNetwork_latency_01_002_0": 0.0021559020454390825,
"DeathStarExpLambdasocialNetwork_packet_loss_01_002_0": 0.0014371869816387733,
"DeathStarExpLambdasocialNetwork_cpu_load_02_002_0": 0.0015847528826406318,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_002_0": 0.001469170311072231,
"DeathStarExpLambdasocialNetwork_crush_02_002_0": 0.0015470732904750578,
"DeathStarExpLambdasocialNetwork_memory_stress_02_002_0": 0.0025543219893926225,
"DeathStarExpLambdasocialNetwork_crush_01_002_0": 0.0015516528592906126,
"DeathStarExpLambdasocialNetwork_memory_stress_01_002_0": 0.0016677909104581856,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_002_0": 0.001513746229648757,
"DeathStarExpLambdasocialNetwork_latency_03_002_0": 0.0014759716648535912,
"DeathStarExpLambdasocialNetwork_packet_loss_02_002_0": 0.001604614837876143,
"DeathStarExpLambdasocialNetwork_cpu_load_01_002_0": 0.0025262431508283447,
"DeathStarExpLambdasocialNetwork_packet_loss_03_002_0": 0.001798384496345182,
"DeathStarExpLambdasocialNetwork_latency_02_002_0": 0.0017879682645329368,
},

"Frequency of an attribute" : {
"DeathStarExpLambdasocialNetwork_latency_01_002_0": 0.021958767789553527,
"DeathStarExpLambdasocialNetwork_packet_loss_01_002_0": 0.020003311811063615,
"DeathStarExpLambdasocialNetwork_cpu_load_02_002_0": 0.02274610650834547,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_002_0": 0.018757497453299084,
"DeathStarExpLambdasocialNetwork_crush_02_002_0": 0.027502475417163177,
"DeathStarExpLambdasocialNetwork_memory_stress_02_002_0": 0.02479345807037275,
"DeathStarExpLambdasocialNetwork_crush_01_002_0": 0.022171243137384368,
"DeathStarExpLambdasocialNetwork_memory_stress_01_002_0": 0.02023646739326452,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_002_0": 0.018450638585237752,
"DeathStarExpLambdasocialNetwork_latency_03_002_0": 0.026811793374646075,
"DeathStarExpLambdasocialNetwork_packet_loss_02_002_0": 0.027420740444747058,
"DeathStarExpLambdasocialNetwork_cpu_load_01_002_0": 0.024211366568985964,
"DeathStarExpLambdasocialNetwork_packet_loss_03_002_0": 0.021233877505677776,
"DeathStarExpLambdasocialNetwork_latency_02_002_0": 0.023448803531673258,
},

"Max value of an attribute for every 5 minute window" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_002_0": 0.00031412580140603496,
"DeathStarExpLambdasocialNetwork_crush_02_002_0": 0.00037594511668042696,
"DeathStarExpLambdasocialNetwork_crush_01_002_0": 0.0004035523141597063,
"DeathStarExpLambdasocialNetwork_packet_loss_02_002_0": 0.0002545476604560529,
"DeathStarExpLambdasocialNetwork_packet_loss_03_002_0": 0.000258304971586083,
},

"Frequency of an attribute after filtering by another attribute" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_002_0": 0.13922052471928856,
"DeathStarExpLambdasocialNetwork_crush_02_002_0": 0.10105176023679299,
"DeathStarExpLambdasocialNetwork_crush_01_002_0": 0.12995665728711606,
"DeathStarExpLambdasocialNetwork_packet_loss_02_002_0": 0.1459112272200306,
"DeathStarExpLambdasocialNetwork_packet_loss_03_002_0": 0.1246977090140202,
},

},


0.05 : {
"Find traces of a service that having an error" : {
"DeathStarExpLambdasocialNetwork_crush_01_005_0": 0.003142603555632328,
"DeathStarExpLambdasocialNetwork_packet_loss_02_005_0": 0.0033731148003744733,
},

"Find traces that have a particular attribute" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_005_0": 0.0324620903623468,
"DeathStarExpLambdasocialNetwork_crush_02_005_0": 0.020292547735835557,
"DeathStarExpLambdasocialNetwork_crush_01_005_0": 0.035856730453632156,
"DeathStarExpLambdasocialNetwork_packet_loss_02_005_0": 0.03040664741474895,
"DeathStarExpLambdasocialNetwork_packet_loss_03_005_0": 0.03135190534284971,
},

"Discover architecture of the whole system" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_005_0": 0.00027817131051946855,
"DeathStarExpLambdasocialNetwork_crush_02_005_0": 0.0002846971709895991,
"DeathStarExpLambdasocialNetwork_crush_01_005_0": 0.0002693899210625753,
"DeathStarExpLambdasocialNetwork_packet_loss_02_005_0": 0.00027930885795537203,
"DeathStarExpLambdasocialNetwork_packet_loss_03_005_0": 0.0002853869259138623,
},

"Find bottlenecks" : {
"DeathStarExpLambdasocialNetwork_latency_01_005_0": 0.01346247361951179,
"DeathStarExpLambdasocialNetwork_cpu_load_02_005_0": 0.014415044873148729,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_005_0": 0.007638246923406554,
"DeathStarExpLambdasocialNetwork_memory_stress_02_005_0": 0.008835479533450074,
"DeathStarExpLambdasocialNetwork_memory_stress_01_005_0": 0.011514552387869966,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_005_0": 0.013406890188299618,
"DeathStarExpLambdasocialNetwork_latency_03_005_0": 0.009571835393557342,
"DeathStarExpLambdasocialNetwork_cpu_load_01_005_0": 0.008049170753279608,
"DeathStarExpLambdasocialNetwork_latency_02_005_0": 0.009234603535207511,
},

"RED metrics - rate" : {
"DeathStarExpLambdasocialNetwork_latency_01_005_0": 0.00013470720695338688,
"DeathStarExpLambdasocialNetwork_packet_loss_01_005_0": 0.00013543766800126143,
"DeathStarExpLambdasocialNetwork_cpu_load_02_005_0": 0.000140376332564098,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_005_0": 0.00013946860360398587,
"DeathStarExpLambdasocialNetwork_crush_02_005_0": 0.00014164920604365307,
"DeathStarExpLambdasocialNetwork_memory_stress_02_005_0": 0.00013611061384258505,
"DeathStarExpLambdasocialNetwork_crush_01_005_0": 0.00013868870809441417,
"DeathStarExpLambdasocialNetwork_memory_stress_01_005_0": 0.00013454400435687787,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_005_0": 0.00013626200508653135,
"DeathStarExpLambdasocialNetwork_latency_03_005_0": 0.00013716943886979316,
"DeathStarExpLambdasocialNetwork_packet_loss_02_005_0": 0.00013337317804626737,
"DeathStarExpLambdasocialNetwork_cpu_load_01_005_0": 0.00013638782088602682,
"DeathStarExpLambdasocialNetwork_packet_loss_03_005_0": 0.00013674737805013994,
"DeathStarExpLambdasocialNetwork_latency_02_005_0": 0.00014028179690311267,
},

"RED metrics - error" : {
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_005_0": 0.0030664130807309207,
"DeathStarExpLambdasocialNetwork_crush_01_005_0": 0.002627742215579159,
"DeathStarExpLambdasocialNetwork_packet_loss_02_005_0": 0.0032617288992756953,
"DeathStarExpLambdasocialNetwork_cpu_load_01_005_0": 0.0037544585433761496,
},

"RED metrics - duration" : {
"DeathStarExpLambdasocialNetwork_latency_01_005_0": 0.0015023114807664467,
"DeathStarExpLambdasocialNetwork_packet_loss_01_005_0": 0.0017615384033917486,
"DeathStarExpLambdasocialNetwork_cpu_load_02_005_0": 0.003297608737977526,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_005_0": 0.001734982875139453,
"DeathStarExpLambdasocialNetwork_crush_02_005_0": 0.0015419116315341218,
"DeathStarExpLambdasocialNetwork_memory_stress_02_005_0": 0.0017385544170873239,
"DeathStarExpLambdasocialNetwork_crush_01_005_0": 0.0016576517609088062,
"DeathStarExpLambdasocialNetwork_memory_stress_01_005_0": 0.001786751024186764,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_005_0": 0.00248120410137491,
"DeathStarExpLambdasocialNetwork_latency_03_005_0": 0.0017863009628309354,
"DeathStarExpLambdasocialNetwork_packet_loss_02_005_0": 0.0015545643824730851,
"DeathStarExpLambdasocialNetwork_cpu_load_01_005_0": 0.0015773649267799173,
"DeathStarExpLambdasocialNetwork_packet_loss_03_005_0": 0.0016164492494065688,
"DeathStarExpLambdasocialNetwork_latency_02_005_0": 0.0015905330075088645,
},

"Frequency of an attribute" : {
"DeathStarExpLambdasocialNetwork_latency_01_005_0": 0.01781367716488476,
"DeathStarExpLambdasocialNetwork_packet_loss_01_005_0": 0.021510570084656327,
"DeathStarExpLambdasocialNetwork_cpu_load_02_005_0": 0.027119293689942335,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_005_0": 0.02157108669121511,
"DeathStarExpLambdasocialNetwork_crush_02_005_0": 0.022300172874547924,
"DeathStarExpLambdasocialNetwork_memory_stress_02_005_0": 0.021061760152205275,
"DeathStarExpLambdasocialNetwork_crush_01_005_0": 0.023088745937010303,
"DeathStarExpLambdasocialNetwork_memory_stress_01_005_0": 0.01965595487555851,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_005_0": 0.025303827935832517,
"DeathStarExpLambdasocialNetwork_latency_03_005_0": 0.02479438182192436,
"DeathStarExpLambdasocialNetwork_packet_loss_02_005_0": 0.020867488712770024,
"DeathStarExpLambdasocialNetwork_cpu_load_01_005_0": 0.028315651644785944,
"DeathStarExpLambdasocialNetwork_packet_loss_03_005_0": 0.022845430782433163,
"DeathStarExpLambdasocialNetwork_latency_02_005_0": 0.025126247908095548,
},

"Max value of an attribute for every 5 minute window" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_005_0": 0.00020295521083359299,
"DeathStarExpLambdasocialNetwork_crush_02_005_0": 0.00022174791442206192,
"DeathStarExpLambdasocialNetwork_crush_01_005_0": 0.0003527621721086109,
"DeathStarExpLambdasocialNetwork_packet_loss_02_005_0": 0.00035081617345862425,
"DeathStarExpLambdasocialNetwork_packet_loss_03_005_0": 0.00028057533645472885,
},

"Frequency of an attribute after filtering by another attribute" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_005_0": 0.1619368269967637,
"DeathStarExpLambdasocialNetwork_crush_02_005_0": 0.06820901124230576,
"DeathStarExpLambdasocialNetwork_crush_01_005_0": 0.1466831133139683,
"DeathStarExpLambdasocialNetwork_packet_loss_02_005_0": 0.08737863838631654,
"DeathStarExpLambdasocialNetwork_packet_loss_03_005_0": 0.16173196826949113,
},

},


0.1 : {
"Find traces of a service that having an error" : {
"DeathStarExpLambdasocialNetwork_packet_loss_03_01_0": 0.0030594035766125685,
},

"Find traces that have a particular attribute" : {
"DeathStarExpLambdasocialNetwork_crush_02_01_0": 0.0372131041583384,
"DeathStarExpLambdasocialNetwork_crush_01_01_0": 0.03165713490538313,
"DeathStarExpLambdasocialNetwork_packet_loss_01_01_0": 0.0326200543041901,
"DeathStarExpLambdasocialNetwork_packet_loss_02_01_0": 0.029914135144506113,
"DeathStarExpLambdasocialNetwork_packet_loss_03_01_0": 0.024642826879183746,
},

"Discover architecture of the whole system" : {
"DeathStarExpLambdasocialNetwork_crush_02_01_0": 0.0002899142431248997,
"DeathStarExpLambdasocialNetwork_crush_01_01_0": 0.00026974561049012866,
"DeathStarExpLambdasocialNetwork_packet_loss_01_01_0": 0.0002715892968180943,
"DeathStarExpLambdasocialNetwork_packet_loss_02_01_0": 0.00028017833688579456,
"DeathStarExpLambdasocialNetwork_packet_loss_03_01_0": 0.0002872415766556269,
},

"Find bottlenecks" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_01_0": 0.007851854632135231,
"DeathStarExpLambdasocialNetwork_memory_stress_02_01_0": 0.007282704737231233,
"DeathStarExpLambdasocialNetwork_latency_01_01_0": 0.008360950376827339,
"DeathStarExpLambdasocialNetwork_latency_03_01_0": 0.007766526499004079,
"DeathStarExpLambdasocialNetwork_latency_02_01_0": 0.008904207331664816,
"DeathStarExpLambdasocialNetwork_cpu_load_02_01_0": 0.011426843108426033,
"DeathStarExpLambdasocialNetwork_cpu_load_01_01_0": 0.008647099112790884,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_01_0": 0.00722257114338905,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_01_0": 0.011639004248750274,
},

"RED metrics - rate" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_01_0": 0.00014108956187590545,
"DeathStarExpLambdasocialNetwork_memory_stress_02_01_0": 0.00014026882209518233,
"DeathStarExpLambdasocialNetwork_latency_01_01_0": 0.00013138670052123067,
"DeathStarExpLambdasocialNetwork_latency_03_01_0": 0.00013876677107880323,
"DeathStarExpLambdasocialNetwork_latency_02_01_0": 0.00014071366350320555,
"DeathStarExpLambdasocialNetwork_crush_02_01_0": 0.00014815717416750298,
"DeathStarExpLambdasocialNetwork_crush_01_01_0": 0.0001322824220393118,
"DeathStarExpLambdasocialNetwork_packet_loss_01_01_0": 0.00013415146568271914,
"DeathStarExpLambdasocialNetwork_packet_loss_02_01_0": 0.00013849334532905016,
"DeathStarExpLambdasocialNetwork_packet_loss_03_01_0": 0.00014125033758152612,
"DeathStarExpLambdasocialNetwork_cpu_load_02_01_0": 0.00014318774372895448,
"DeathStarExpLambdasocialNetwork_cpu_load_01_01_0": 0.00013251909141425862,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_01_0": 0.00013736186264490079,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_01_0": 0.00014424392769121224,
},

"RED metrics - error" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_01_0": 0.0022884816673369992,
"DeathStarExpLambdasocialNetwork_memory_stress_02_01_0": 0.002681133937191741,
"DeathStarExpLambdasocialNetwork_latency_03_01_0": 0.0023585328628900554,
"DeathStarExpLambdasocialNetwork_latency_02_01_0": 0.002835886454131077,
"DeathStarExpLambdasocialNetwork_packet_loss_03_01_0": 0.0023272234168816544,
"DeathStarExpLambdasocialNetwork_cpu_load_01_01_0": 0.004098653686238397,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_01_0": 0.003148367374781306,
},

"RED metrics - duration" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_01_0": 0.0014033621908512328,
"DeathStarExpLambdasocialNetwork_memory_stress_02_01_0": 0.0018045390880414192,
"DeathStarExpLambdasocialNetwork_latency_01_01_0": 0.001408979772737455,
"DeathStarExpLambdasocialNetwork_latency_03_01_0": 0.0016050434053018353,
"DeathStarExpLambdasocialNetwork_latency_02_01_0": 0.0017903718292224713,
"DeathStarExpLambdasocialNetwork_crush_02_01_0": 0.0013411648134666156,
"DeathStarExpLambdasocialNetwork_crush_01_01_0": 0.001592962824927423,
"DeathStarExpLambdasocialNetwork_packet_loss_01_01_0": 0.0012098288838373102,
"DeathStarExpLambdasocialNetwork_packet_loss_02_01_0": 0.0014527265088596448,
"DeathStarExpLambdasocialNetwork_packet_loss_03_01_0": 0.0014712894216841494,
"DeathStarExpLambdasocialNetwork_cpu_load_02_01_0": 0.0017035013108843236,
"DeathStarExpLambdasocialNetwork_cpu_load_01_01_0": 0.0018167983141341645,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_01_0": 0.00142463013176912,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_01_0": 0.001760246899284739,
},

"Frequency of an attribute" : {
"DeathStarExpLambdasocialNetwork_memory_stress_01_01_0": 0.022091782573060464,
"DeathStarExpLambdasocialNetwork_memory_stress_02_01_0": 0.028608117284504374,
"DeathStarExpLambdasocialNetwork_latency_01_01_0": 0.021384165368061377,
"DeathStarExpLambdasocialNetwork_latency_03_01_0": 0.019864502269004068,
"DeathStarExpLambdasocialNetwork_latency_02_01_0": 0.029494534428037174,
"DeathStarExpLambdasocialNetwork_crush_02_01_0": 0.030274887573390797,
"DeathStarExpLambdasocialNetwork_crush_01_01_0": 0.023034098789370308,
"DeathStarExpLambdasocialNetwork_packet_loss_01_01_0": 0.023015367351946033,
"DeathStarExpLambdasocialNetwork_packet_loss_02_01_0": 0.0244689364151411,
"DeathStarExpLambdasocialNetwork_packet_loss_03_01_0": 0.022047131554906677,
"DeathStarExpLambdasocialNetwork_cpu_load_02_01_0": 0.0250057161243449,
"DeathStarExpLambdasocialNetwork_cpu_load_01_01_0": 0.019208372024067416,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_01_0": 0.020235689908565634,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_01_0": 0.030635648795331254,
},

"Max value of an attribute for every 5 minute window" : {
"DeathStarExpLambdasocialNetwork_crush_02_01_0": 0.0002905684313575578,
"DeathStarExpLambdasocialNetwork_crush_01_01_0": 0.00021590621680220374,
"DeathStarExpLambdasocialNetwork_packet_loss_01_01_0": 0.00025129179874771245,
"DeathStarExpLambdasocialNetwork_packet_loss_02_01_0": 0.00022612291606919274,
"DeathStarExpLambdasocialNetwork_packet_loss_03_01_0": 0.00024222388510683742,
},

"Frequency of an attribute after filtering by another attribute" : {
"DeathStarExpLambdasocialNetwork_crush_02_01_0": 0.1542040625809913,
"DeathStarExpLambdasocialNetwork_crush_01_01_0": 0.14733593648577473,
"DeathStarExpLambdasocialNetwork_packet_loss_01_01_0": 0.16584434643470414,
"DeathStarExpLambdasocialNetwork_packet_loss_02_01_0": 0.15437852149057568,
"DeathStarExpLambdasocialNetwork_packet_loss_03_01_0": 0.10640362410826702,
},

},


0.2 : {
"Find traces of a service that having an error" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_02_0": 0.07542438555519422,
},

"Find traces that have a particular attribute" : {
"DeathStarExpLambdasocialNetwork_crush_02_02_0": 0.03823019237656331,
"DeathStarExpLambdasocialNetwork_crush_01_02_0": 0.02984670203312083,
"DeathStarExpLambdasocialNetwork_packet_loss_01_02_0": 0.030440490034144296,
"DeathStarExpLambdasocialNetwork_packet_loss_03_02_0": 0.030442570675796043,
"DeathStarExpLambdasocialNetwork_packet_loss_02_02_0": 0.03126205892430489,
},

"Discover architecture of the whole system" : {
"DeathStarExpLambdasocialNetwork_crush_02_02_0": 0.00028905081420479306,
"DeathStarExpLambdasocialNetwork_crush_01_02_0": 0.00026103017616911444,
"DeathStarExpLambdasocialNetwork_packet_loss_01_02_0": 0.00030165770503228384,
"DeathStarExpLambdasocialNetwork_packet_loss_03_02_0": 0.00030607052761262257,
"DeathStarExpLambdasocialNetwork_packet_loss_02_02_0": 0.0002845200128624896,
},

"Find bottlenecks" : {
"DeathStarExpLambdasocialNetwork_cpu_load_02_02_0": 0.010138192627785791,
"DeathStarExpLambdasocialNetwork_cpu_load_01_02_0": 0.007225441219694191,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_02_0": 0.005737018555759144,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_02_0": 0.008585696218689114,
"DeathStarExpLambdasocialNetwork_memory_stress_01_02_0": 0.007523589983858951,
"DeathStarExpLambdasocialNetwork_memory_stress_02_02_0": 0.011797197793977313,
"DeathStarExpLambdasocialNetwork_latency_01_02_0": 0.006160125091861032,
},

"RED metrics - rate" : {
"DeathStarExpLambdasocialNetwork_crush_02_02_0": 0.0001469347781686108,
"DeathStarExpLambdasocialNetwork_crush_01_02_0": 0.00012768019630399595,
"DeathStarExpLambdasocialNetwork_packet_loss_01_02_0": 0.0001442361044275405,
"DeathStarExpLambdasocialNetwork_packet_loss_03_02_0": 0.0001545452179428567,
"DeathStarExpLambdasocialNetwork_packet_loss_02_02_0": 0.00014725805666557213,
"DeathStarExpLambdasocialNetwork_cpu_load_02_02_0": 0.00015467510636196015,
"DeathStarExpLambdasocialNetwork_cpu_load_01_02_0": 0.00013602668023926693,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_02_0": 0.0001461574389859756,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_02_0": 0.00015065977214236736,
"DeathStarExpLambdasocialNetwork_memory_stress_01_02_0": 0.00014602230859925473,
"DeathStarExpLambdasocialNetwork_memory_stress_02_02_0": 0.00015933930557432223,
"DeathStarExpLambdasocialNetwork_latency_01_02_0": 0.00013219643001499343,
},

"RED metrics - error" : {
"DeathStarExpLambdasocialNetwork_packet_loss_01_02_0": 0.002728986446914902,
"DeathStarExpLambdasocialNetwork_cpu_load_01_02_0": 0.003137292795494348,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_02_0": 0.002609660556966082,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_02_0": 0.002735541474407544,
"DeathStarExpLambdasocialNetwork_memory_stress_01_02_0": 0.002016131108924718,
"DeathStarExpLambdasocialNetwork_latency_01_02_0": 0.008558391704276943,
},

"RED metrics - duration" : {
"DeathStarExpLambdasocialNetwork_crush_02_02_0": 0.0016361396012883422,
"DeathStarExpLambdasocialNetwork_crush_01_02_0": 0.0016545709969497818,
"DeathStarExpLambdasocialNetwork_packet_loss_01_02_0": 0.0013099495576180777,
"DeathStarExpLambdasocialNetwork_packet_loss_03_02_0": 0.0019125664423840674,
"DeathStarExpLambdasocialNetwork_packet_loss_02_02_0": 0.0018876043195948098,
"DeathStarExpLambdasocialNetwork_cpu_load_02_02_0": 0.0015069375242880289,
"DeathStarExpLambdasocialNetwork_cpu_load_01_02_0": 0.0016228793184946896,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_02_0": 0.0012824581363675934,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_02_0": 0.001975741834417151,
"DeathStarExpLambdasocialNetwork_memory_stress_01_02_0": 0.0013187477217019082,
"DeathStarExpLambdasocialNetwork_memory_stress_02_02_0": 0.001827892188832572,
"DeathStarExpLambdasocialNetwork_latency_01_02_0": 0.001157531086499614,
},

"Frequency of an attribute" : {
"DeathStarExpLambdasocialNetwork_crush_02_02_0": 0.0360135157372828,
"DeathStarExpLambdasocialNetwork_crush_01_02_0": 0.022890160376492,
"DeathStarExpLambdasocialNetwork_packet_loss_01_02_0": 0.02294900429589613,
"DeathStarExpLambdasocialNetwork_packet_loss_03_02_0": 0.028471014506900816,
"DeathStarExpLambdasocialNetwork_packet_loss_02_02_0": 0.024838301359415546,
"DeathStarExpLambdasocialNetwork_cpu_load_02_02_0": 0.027327791570856234,
"DeathStarExpLambdasocialNetwork_cpu_load_01_02_0": 0.024554773241449376,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_02_0": 0.02423219076594067,
"DeathStarExpLambdasocialNetwork_disk_io_stress_02_02_0": 0.03373446571061953,
"DeathStarExpLambdasocialNetwork_memory_stress_01_02_0": 0.027087268805488996,
"DeathStarExpLambdasocialNetwork_memory_stress_02_02_0": 0.03156066975218286,
"DeathStarExpLambdasocialNetwork_latency_01_02_0": 0.017923573011380166,
},

"Max value of an attribute for every 5 minute window" : {
"DeathStarExpLambdasocialNetwork_crush_02_02_0": 0.0006123458490997835,
"DeathStarExpLambdasocialNetwork_crush_01_02_0": 0.00024058433195450548,
"DeathStarExpLambdasocialNetwork_packet_loss_01_02_0": 0.00021583856159000173,
"DeathStarExpLambdasocialNetwork_packet_loss_03_02_0": 0.0002734254809485254,
"DeathStarExpLambdasocialNetwork_packet_loss_02_02_0": 0.0002212733854476453,
},

"Frequency of an attribute after filtering by another attribute" : {
"DeathStarExpLambdasocialNetwork_crush_02_02_0": 0.15435346924272805,
"DeathStarExpLambdasocialNetwork_crush_01_02_0": 0.1463087702024002,
"DeathStarExpLambdasocialNetwork_packet_loss_01_02_0": 0.16259295115446482,
"DeathStarExpLambdasocialNetwork_packet_loss_03_02_0": 0.14674332195582834,
"DeathStarExpLambdasocialNetwork_packet_loss_02_02_0": 0.13682678411433788,
},}}
    query_to_data = defaultdict(dict)
    for lambda_exp, lambda_data in raw_data.items():
        for query, query_data in lambda_data.items():
            incidents_emds = list(query_data.values())
            query_to_data[query][lambda_exp] = (numpy.average(incidents_emds), numpy.std(incidents_emds))

    query_to_data = {
        'Find traces of a service that having an error': {0.001: (0.0018146608989544966, 0.0012781242567446681), 0.002: (0.0027856469024895043, 0.0), 0.005: (0.0018517588112684326, 0.0007697905806104517), 0.01: (0.0008015431775271952, 0.0), 0.02: (0.003294520529182275, 0.00206649460333428), 0.05: (0.0092578591780034003, 0.00011525562237107274), 0.1: (0.0230594035766125685, 0.0), 0.2: (0.07542438555519422, 0.0)},
        'Find traces that have a particular attribute': {0.001: (0.026563872964080894, 0.006510588159685985), 0.002: (0.0305960265990617, 0.0023432414436482213), 0.005: (0.02999656179810679, 0.0018425842430695798), 0.01: (0.0291424504469714, 0.0041724493254400846), 0.02: (0.03544134114203268, 0.0049794607851545635), 0.05: (0.040073984261882635, 0.005226154453148332), 0.1: (0.045209451078320294, 0.004075106607590468), 0.2: (0.04704440280878588, 0.0031255190161221195)},
        'Discover architecture of the whole system': {0.001: (0.00028126465844191225, 4.501398954574934e-07), 0.002: (0.00028144701464720634, 8.645759043375735e-07), 0.005: (0.0002800126230141474, 1.8037111873393155e-06), 0.01: (0.0002778432516375627, 2.2882218019846616e-06), 0.02: (0.0002785089412774594, 4.970274823783444e-06), 0.05: (0.00027939083728817544, 5.755389607316383e-06), 0.1: (0.0002797338127949088, 8.07839844693592e-06), 0.2: (0.00028846584717626074, 1.5827682871655668e-05)},
        'Find bottlenecks': {0.001: (0.009743880645575298, 0.0023935321514104956), 0.002: (0.008805455241480385, 0.0013165978076394647), 0.005: (0.008106498455951605, 0.0015124053886575687), 0.01: (0.008499878433572514, 0.001593258326691389), 0.02: (0.00936118521680677, 0.00220996667641375), 0.05: (0.020680921911970134, 0.0024202410251152025), 0.1: (0.050789084576690994, 0.0015614698834167282), 0.2: (0.058166751641660792, 0.002016790540612222)},
        'RED metrics - rate': {0.001: (0.00014247471058641254, 4.1493684022863437e-07), 0.002: (0.00014254654228268063, 4.079878519490556e-07), 0.005: (0.00014244176482321926, 4.612111807711231e-07), 0.01: (0.00014206894075792737, 5.848846698014287e-07), 0.02: (0.00013976718787671755, 1.6337915879005049e-06), 0.05: (0.00013722885437872384, 2.400653121407608e-06), 0.1: (0.00013884806352526885, 4.736912050210325e-06), 0.2: (0.00014547761628555972, 9.017890277583852e-06)},
        'RED metrics - error': {0.001: (0.004263686250395129, 0.0013810966212709963), 0.002: (0.004790873284345922, 0.0014129218598996315), 0.005: (0.0037374968735541837, 0.0008743884128652819), 0.01: (0.004036102364349881, 0.0009902431881149767), 0.02: (0.005460022244215845, 0.0027002722236062733), 0.05: (0.0031775856847404815, 0.00040452118734512273), 0.1: (0.0028197541999216043, 0.0005976870346250117), 0.2: (0.0036310006811640892, 0.0022282035374347423)},
        'RED metrics - duration': {0.001: (0.0018687096169533045, 0.00027701239674331635), 0.002: (0.0018503175822870966, 0.0003610911464847691), 0.005: (0.0017641404264640774, 0.00022172798958346472), 0.01: (0.0017618015070661975, 0.0002504741355098833), 0.02: (0.0017624842796065822, 0.00036441653051093905), 0.05: (0.0018305519258118906, 0.00046801341218123986), 0.1: (0.0015561032425001362, 0.00018859699437670182), 0.2: (0.0015910848940363861, 0.0002655009230808104)},
        'Frequency of an attribute': {0.001: (0.01968968781077352, 0.004035456498685029), 0.002: (0.020767711325732508, 0.0026053703712723966), 0.005: (0.019520338506387024, 0.0017997697119031225), 0.01: (0.0205118717096186, 0.002890602098722501), 0.02: (0.019839039113672462, 0.00291712958421251), 0.05: (0.022955306448275866, 0.002795555245371632), 0.1: (0.024240639318552252, 0.0038329723990398546), 0.2: (0.026798560761158757, 0.004857845323560741)},
        'Max value of an attribute for every 5 minute window': {0.001: (0.0003024788070640252, 5.801899823704601e-05), 0.002: (0.00028351796399322796, 4.985132864231307e-05), 0.005: (0.0002624059544595192, 4.2965232622088665e-05), 0.01: (0.0002893373063757069, 3.833878273378697e-05), 0.02: (0.00032129517285766084, 6.037756108036204e-05), 0.05: (0.00028177136145552377, 6.26468060321772e-05), 0.1: (0.00024522264961670084, 2.5792712676266684e-05), 0.2: (0.0003126935218080923, 0.00015117655651365175)},
        'Attribute frequency after filtering by another attribute': {0.001: (0.10809871492400372, 0.025656935989546444), 0.002: (0.1202706058053867, 0.01421091375114189), 0.005: (0.12333521198044219, 0.014949352280561636), 0.01: (0.12222837574151006, 0.024581827682515447), 0.02: (0.12816757569544968, 0.015411119928582757), 0.05: (0.12518791164176907, 0.03955788362474583), 0.1: (0.14563329822006257, 0.020495315121743338), 0.2: (0.1493650593339519, 0.008640417941875807)}
    }
    print(query_to_data)
    for query, data in query_to_data.items():
        print(query, "decay rate", data[min(data)][0] / data[max(data)][0])
    fig, ax = pyplot.subplots()
    for index, (query, data) in enumerate(query_to_data.items()):
        lambdas = sorted(list(data.keys()))
        lambda_to_e = {0.2: 4.5, 0.1: 9, 0.05: 20, 0.02: 50, 0.01: 100, 0.005: 200, 0.002: 500, 0.001: 1000}
        ax.errorbar([lambda_to_e[k] for k in lambdas], [data[k][0] for k in lambdas],
                    yerr=[data[k][1] for k in lambdas], capsize=5, label=f"Q{index + 1}")
    ax.set_ylabel("EMD")
    ax.set_xlabel("inter-arrival count of traces w/ incidents", fontsize=21)
    ax.set_xscale('log')
    fig.legend(ncol=2, fontsize=12)
    fig.tight_layout()
    set_size(ax)
    fig.savefig(r"figures/exp_lambda_by_query.pdf", format="pdf", bbox_inches='tight')


def exp_lambda_graph():
    raw_data = {
"DeathStarExpLambdasocialNetwork_memory_stress_01_001_0": 0.009014899209649825,
"DeathStarExpLambdasocialNetwork_memory_stress_01_001_1": 0.008444451452499998,
"DeathStarExpLambdasocialNetwork_memory_stress_01_001_2": 0.0073802780592580955,
"DeathStarExpLambdasocialNetwork_latency_01_002_0": 0.00956217704737734,
"DeathStarExpLambdasocialNetwork_latency_01_002_1": 0.007050177425525647,
"DeathStarExpLambdasocialNetwork_latency_01_002_2": 0.00680411422223681,
"DeathStarExpLambdasocialNetwork_latency_01_005_0": 0.008228292368029095,
"DeathStarExpLambdasocialNetwork_latency_01_005_1": 0.008219799329165636,
"DeathStarExpLambdasocialNetwork_latency_01_005_2": 0.00615737644582798,
"DeathStarExpLambdasocialNetwork_crush_01_001_0": 0.030949271718409974,
"DeathStarExpLambdasocialNetwork_crush_01_001_1": 0.0245284859340283,
"DeathStarExpLambdasocialNetwork_crush_01_001_2": 0.028630871949340886,
"DeathStarExpLambdasocialNetwork_crush_02_02_0": 0.033040235485619386,
"DeathStarExpLambdasocialNetwork_crush_02_02_1": 0.026307334109100352,
"DeathStarExpLambdasocialNetwork_crush_02_02_2": 0.017721867133801045,
"DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.02095662326151025,
"DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.025056265235604874,
"DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.02900147461976445,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.00945805382142407,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.007448498477591209,
"DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.009462696434624513,
"DeathStarExpLambdasocialNetwork_crush_01_02_0": 0.02876135690191292,
"DeathStarExpLambdasocialNetwork_crush_01_02_1": 0.030715448593535373,
"DeathStarExpLambdasocialNetwork_crush_01_02_2": 0.023856186569106222,
"DeathStarExpLambdasocialNetwork_latency_02_0005_0": 0.00853001161879961,
"DeathStarExpLambdasocialNetwork_latency_02_0005_1": 0.0064151156598223295,
"DeathStarExpLambdasocialNetwork_latency_02_0005_2": 0.008208964218847599,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0002_0": 0.006284127264267041,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0002_1": 0.008739183283522203,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0002_2": 0.009662773166461414,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0005_0": 0.0062748589146202835,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0005_1": 0.006953670724337641,
"DeathStarExpLambdasocialNetwork_memory_stress_01_0005_2": 0.007251468520901995,
"DeathStarExpLambdasocialNetwork_latency_02_0002_0": 0.006206794937231682,
"DeathStarExpLambdasocialNetwork_latency_02_0002_1": 0.009859593409023586,
"DeathStarExpLambdasocialNetwork_latency_02_0002_2": 0.006249725278246959,
        "DeathStarExpLambdasocialNetwork_packet_loss_01_005_0": 0.031183941433787558,
        "DeathStarExpLambdasocialNetwork_packet_loss_01_005_1": 0.021308664544222485,
        "DeathStarExpLambdasocialNetwork_packet_loss_01_005_2": 0.02491982897178682,
        "DeathStarExpLambdasocialNetwork_packet_loss_01_002_0": 0.022387290921521086,
        "DeathStarExpLambdasocialNetwork_packet_loss_01_002_1": 0.02696279308557238,
        "DeathStarExpLambdasocialNetwork_packet_loss_01_002_2": 0.017562265089407374,
        "DeathStarExpLambdasocialNetwork_packet_loss_01_02_0": 0.032900833268364696,
        "DeathStarExpLambdasocialNetwork_packet_loss_01_02_1": 0.030957072477555393,
        "DeathStarExpLambdasocialNetwork_packet_loss_01_02_2": 0.036032271827998195,
        "DeathStarExpLambdasocialNetwork_packet_loss_02_0002_0": 0.020800974342203443,
        "DeathStarExpLambdasocialNetwork_packet_loss_02_0002_1": 0.02786826295601386,
        "DeathStarExpLambdasocialNetwork_packet_loss_02_0002_2": 0.014796629907306982,
        "DeathStarExpLambdasocialNetwork_cpu_load_02_002_0": 0.007364475723303553,
        "DeathStarExpLambdasocialNetwork_cpu_load_02_002_1": 0.006484990077074751,
        "DeathStarExpLambdasocialNetwork_cpu_load_02_002_2": 0.007164404618219475,
        "DeathStarExpLambdasocialNetwork_cpu_load_02_005_0": 0.011243080908408172,
        "DeathStarExpLambdasocialNetwork_cpu_load_02_005_1": 0.009452408835782189,
        "DeathStarExpLambdasocialNetwork_cpu_load_02_005_2": 0.0062829438066914585,
        "DeathStarExpLambdasocialNetwork_packet_loss_02_0005_0": 0.02755443895390622,
        "DeathStarExpLambdasocialNetwork_packet_loss_02_0005_1": 0.024574750803587245,
        "DeathStarExpLambdasocialNetwork_packet_loss_02_0005_2": 0.027963421127954198,
        "DeathStarExpLambdasocialNetwork_packet_loss_03_02_0": 0.02975764497248761,
        "DeathStarExpLambdasocialNetwork_packet_loss_03_02_1": 0.023541765205624225,
        "DeathStarExpLambdasocialNetwork_packet_loss_03_02_2": 0.02585134148004048,
        "DeathStarExpLambdasocialNetwork_disk_io_stress_02_002_0": 0.1061977853650460515,
        "DeathStarExpLambdasocialNetwork_disk_io_stress_02_002_1": 0.11076090301167204,
        "DeathStarExpLambdasocialNetwork_disk_io_stress_02_002_2": 0.109422980953746204,
        "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.006792846890662989,
        "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.007769582542151393,
        "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.007012414671046627,
        "DeathStarExpLambdasocialNetwork_disk_io_stress_02_005_0": 0.006830039634819205,
        "DeathStarExpLambdasocialNetwork_disk_io_stress_02_005_1": 0.00806272474699636,
        "DeathStarExpLambdasocialNetwork_disk_io_stress_02_005_2": 0.007111340320120401,
        "DeathStarExpLambdasocialNetwork_packet_loss_02_02_0": 0.02792397145323269,
        "DeathStarExpLambdasocialNetwork_packet_loss_02_02_1": 0.02491516277649162,
        "DeathStarExpLambdasocialNetwork_packet_loss_02_02_2": 0.03263437944225765,
        "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.020030439904527866,
    }
    data = defaultdict(list)
    for k, v in raw_data.items():
        data[k.split('_')[-2]].append(v)
    data = {float(f"0.{k[1:]}"): v for k, v in data.items()}
    keys = sorted(list(data.keys()))
    data = {k: data[k] for k in keys}
    print(pprint(data))
    data = {0.001: [0.02095662326151025, 0.025056265235604874, 0.02900147461976445, 0.00945805382142407, 0.007448498477591209, 0.009462696434624513, 0.006792846890662989, 0.007769582542151393, 0.007012414671046627, 0.020030439904527866],
 0.002: [0.006284127264267041, 0.008739183283522203, 0.009662773166461414, 0.006206794937231682, 0.009859593409023586, 0.006249725278246959, 0.020800974342203443, 0.02786826295601386, 0.014796629907306982],
 0.005: [0.00853001161879961, 0.0064151156598223295, 0.008208964218847599, 0.0062748589146202835, 0.006953670724337641, 0.007251468520901995, 0.02755443895390622, 0.024574750803587245, 0.027963421127954198],
 0.01: [0.009014899209649825, 0.008444451452499998, 0.0073802780592580955, 0.010949271718409974, 0.0245284859340283, 0.018630871949340886],
 0.02: [0.01956217704737734, 0.017050177425525647, 0.01680411422223681, 0.022387290921521086, 0.02696279308557238, 0.027562265089407374, 0.007364475723303553, 0.006484990077074751, 0.007164404618219475, 0.00619778536504605, 0.01076090301167205, 0.00942298095374621],
 0.05: [0.028228292368029095, 0.028219799329165636, 0.02615737644582798, 0.031183941433787558, 0.021308664544222485, 0.02491982897178682, 0.011243080908408172, 0.009452408835782189, 0.0062829438066914585, 0.006830039634819205, 0.00806272474699636, 0.007111340320120401],
 0.2: [0.033040235485619386, 0.026307334109100352, 0.017721867133801045, 0.02876135690191292, 0.030715448593535373, 0.023856186569106222, 0.032900833268364696, 0.030957072477555393, 0.036032271827998195, 0.02975764497248761, 0.023541765205624225, 0.02585134148004048, 0.02792397145323269, 0.02491516277649162, 0.03263437944225765]}
    lambda_to_e = {0.2: 4.5, 0.1: 9, 0.05: 20, 0.02: 50, 0.01: 100, 0.005: 200, 0.002: 500, 0.001: 1000}

    fig, ax = pyplot.subplots()
    ax.errorbar([lambda_to_e[k] for k in keys], [numpy.average(data[k]) for k in keys], yerr=[numpy.std(data[k]) for k in keys], capsize=5)
    ax.set_ylabel("EMD")
    ax.set_xlabel("inter-arrival count of traces w/ incidents", fontsize=21)
    ax.set_xscale('log')
    fig.tight_layout()
    # ax.set_ylim(0, 1)
    fig.savefig(r"figures/exp_lambda.pdf", format="pdf", bbox_inches='tight')


def exp_lambda_specific():
    exp_02 = {'real': {0.0: 0.3225547255038221, 1.0: 0.3550479933981932, 5.0: 0.00964754169562196, 6.0: 0.03192863968033356, 7.0: 0.05887334954829743, 8.0: 0.031163134120917303, 9.0: 0.016580524669909658, 10.0: 0.010700790479499652, 11.0: 0.014278578874218207, 12.0: 0.008301120569840167}, 'syn': {0.0: 0.27973732313316635, 1.0: 0.4448243179202491, 2.0: 0.01773745853361316, 3.0: 0.012152190102227338, 7.0: 0.0786676596032767, 8.0: 0.03134520343917135, 9.0: 0.010324284070137432, 20.0: 0.010696635298896487, 60.0: 0.01851601110283664, 75.0: 0.0189899126667118, 168.0: 0.020276216911515807, 545.0: 0.01309999322997766, 1514.0: 0.007108523458127412}}
    exp_0001 = {'real': {0.0: 0.4038831721951068, 1.0: 0.43713464109180217, 5.0: 0.0009643423384213635, 6.0: 0.0031821431902260082, 7.0: 0.0059203531569621045, 8.0: 0.0031069108092144125, 9.0: 0.0016426772779556688, 10.0: 0.0010339789224982124, 11.0: 0.0014723163490533778, 12.0: 0.0008070382690334814}, 'syn': {0.0: 0.3537537957623938, 1.0: 0.465287113173428, 2.0: 0.010671261387287181, 7.0: 0.009996929270872429, 75.0: 0.10870381111603944, 204.0: 0.020198573816916306, 561.0: 0.01593367225084445, 660.0: 0.007506226756286465}}

    for data in [exp_02, exp_0001]:
        pretty = 1000 if data == exp_0001 else 4.5
        fig, ax = pyplot.subplots()
        ax.bar(list(data['syn'].keys()), list(data['syn'].values()), label="Synthetic", color='blue', alpha=0.7)
        ax.bar(list(data['real'].keys()), list(data['real'].values()), label="Real", color='orange', alpha=0.7)
        ax.set_xlim(min(data['real']), max(data['real']))
        ax.set_ylim(0, 0.55)
        ax.set_title(f"Find bottlenecks (~{pretty} inter-arrival count of incidents)", fontsize=18)
        ax.set_ylabel("% of traces")
        ax.set_xlabel("Rounded ratio")
        ax.legend(fontsize=20)
        fig.tight_layout()
        fig.savefig(f"figures/exp_lambda_{pretty}.pdf", format="pdf", bbox_inches='tight')


def exp_lambda_application():
    raw_data = {
    5: {
        "Find traces of a service that having an error": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.003744123968416018,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0037408888520755134,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.05143367575269947,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.003838820771693801,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0037920547661877764,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.003799083580334346,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.003810592018423724,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0037995834454556295,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0038485151132981155,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.003854581310472711,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.003750173024954518,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0037966890323148685,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.058416435587553266,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0038338069608817465,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0038072080729336954,
        },

        "Find traces that have a particular attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0056275917643552826,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0062097503284895125,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.007918219014099113,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.007198452252251952,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.007683008957463521,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.005691739182933233,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.007568860277955138,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0058852269653850734,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.006052912467231671,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.008191794579901575,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.006658531738678587,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.007462284400576814,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.006480436430262118,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.006144382656474126,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.005555040188075154,
        },

        "Discover architecture of the whole system": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0035144342522856835,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.008633947560371301,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.009119790285244503,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.008077077317656252,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.008163179041231122,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.008320024307308663,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.008657071391300324,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0030671219718253068,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.008098619007559595,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.008962600471712044,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.008617018245305091,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.009238185367715236,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.008432362614970152,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.008970481754068982,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.007968072055372217,
        },

        "Find bottlenecks": {
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0025858116121875406,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0021687536382324598,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.00231683690307007,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.002361148714271551,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0026032885511011123,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.002287779245248221,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.003134604788961734,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0024969669445197894,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.002820476127578454,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0025109302565508073,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0022712569275931886,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.002457063197207652,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.002949909631932745,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.00218947673494632,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0020774140444139956,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.002089981614821727,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.00241239232305841,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0027056831808230323,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0023757064065716284,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.002255094014385957,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.002460908753580101,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0024892832822797422,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0030210808492726277,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.002836188859651699,
        },

        "RED metrics - rate": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0005285733868211675,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0005290354743898287,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0005304247014723606,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0005367562787362603,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0005465864217585073,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0005306013548993457,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0005111448408403054,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.000523305800343465,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0005332655937739522,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0005339181725799154,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0005306737624990197,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0005402593342738488,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0005251072331894304,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.000537958551622339,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0005320561354562214,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0005395638634950345,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0005197914420275504,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.0005375881619046349,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0005450353928465891,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0005291669659385352,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0005326124392984958,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0005272038990281775,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0005327572544308905,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0005426807145874143,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0005343541819438764,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.0005184561993495228,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0005380270379235077,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0005281548895796168,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0005291195999859809,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0005196900665670639,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.000545030834603637,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0005180335507918788,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0005375988212268672,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.00052483540641423,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.000551721422921453,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0005330692505731328,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0005283388850357152,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0005221215131626762,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0005157271727469513,
        },

        "RED metrics - error": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.00553035318874284,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.005758043352838242,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.005049213818284875,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.006612970457485515,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0059786131201642586,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.00597591309925238,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.005665783725025457,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.005849290397922621,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.005852080311137383,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.00563094694711463,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.005573092246432027,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.005713953863866649,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.005709075567994614,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.005240705289419333,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.005649530700966475,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.005510983536218644,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.006067354758557082,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.005727526218927519,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.005598850464802744,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0051884729488189205,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.006256572073450095,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.00592572717685919,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.005832495422847076,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.005539624823862378,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.00620931050473244,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.006620458461763029,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.006118629529652073,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.005627412774348452,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.005683696388262874,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.00575177512710739,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.005672669190760084,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.00623191432291825,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.005482609550582377,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.005654435673541881,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0062052591484837395,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.006031432229606505,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.00609958778327126,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.005454748621814925,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0059360658390976125,
        },

        "RED metrics - duration": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0006538117484969239,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0005305780551526806,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0006053220793289213,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0005486258171150612,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.000489839306138966,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0006363511414938837,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0006530409681664139,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0007134978525990385,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0006540957608219061,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0006848825415150656,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0006078655075474154,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0006195007971834732,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0007022851655119585,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0006265476637646548,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0005679992146567627,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0006175519811142351,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0005139742107943869,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.0006158188866193394,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0006226923730638369,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0006239870738147815,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0005774222562047939,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0007730849012980576,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0007571043612226115,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.000730161451619744,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.000513528147098691,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.000480014503794486,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0005251249052176015,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0008989284379148002,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0007015024480715069,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0005498789745612984,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0005580212149545199,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0005749279679132196,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0007073181817882001,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.000565590898203816,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0006106192108844848,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0005557662843344355,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0006372980456749636,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0007255916361154088,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0006441710813686631,
        },

        "Frequency of an attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.000731300620427539,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0023988722501312397,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.002449701893002873,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.003717754608993197,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0006689873860299202,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0017718576695847934,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.001287148008615408,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0016954018275923553,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0014942476939555315,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0018302244057602842,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0022205729937159395,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0022174753154983197,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0008082862716403131,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0012312223195097528,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0012747489123133358,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0027340121848712477,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0016346800406229246,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.0013537757050386646,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0018268383609754096,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0020248309016054683,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0025226109960166804,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0014928393403545856,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0005722134807103828,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.001468902231502354,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0018266481433106465,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.0017703086101095598,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0029441465245490447,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0012008875277443487,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.002727794065138687,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0012644431056186426,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0021059047118871937,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0016444442318350504,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0023655423268153656,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0014998006327806637,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0013697515425552592,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0010850824345918053,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0008110390916184136,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0011384072053758634,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0014615418010700126,
        },

        "Max value of an attribute for every 5 minute window": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0009259794479943386,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0009286485602388955,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0009578765253237942,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0009553725456882481,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0009661810217826014,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0009568978131498654,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0009500414527921675,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0009444141654523087,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.000926664379684272,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0009530611935281936,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0009185899973365674,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.000922027498333148,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0009637361212939816,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0009644979591235499,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.000943157148196113,
        },

        "Frequency of an attribute after filtering by another attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.017562052354275947,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.018271912751359608,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0076698679139080754,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.014880218677008727,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.012651236937959966,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0068654254199493735,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.022890589334568674,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.01712049877811612,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.01389802278685955,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.008060835831156321,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.006559571426670528,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.017440166609090747,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.01826412076989867,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.009093348053932906,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.01670005558635166,
        },

    },

    10: {
        "Find traces of a service that having an error": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.003902848471689853,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.003906576953779227,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.003925308270255712,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.003955605396121155,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.003964449554906153,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0039564581728942895,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.00392951771123834,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0039408981596280096,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.003982980065686093,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0031591840272927746,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.003947394462046926,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.003987554365295364,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.003953742571451758,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.00398089876839015,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0039751620853780935,
        },

        "Find traces that have a particular attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.010218149553436771,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.015527538397241317,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.014250074190283702,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.008547365562170206,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.008921073488297489,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.011993772462536164,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.02492915655763864,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.015619961890431503,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.01690840953215443,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.007558828128572899,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.015067609872005996,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.014477985060230245,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.013683624997891178,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.01711773440008697,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0131777774983533,
        },

        "Discover architecture of the whole system": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0012259565659889084,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.001224477838467434,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.003669406585885201,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.006269700831090664,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0012259395787268222,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0012353020855037236,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0012347389590381037,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0012235954705485253,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.009096706570310696,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.009317017107736076,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0035663875146513795,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0036397520797968887,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0012318110855902326,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0012337895538492084,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.00865920326870943,
        },

        "Find bottlenecks": {
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0052427411616844865,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0047284559785152076,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.00549381755214051,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.005356306390010319,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.004909379089929439,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.005533063701158469,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0046114876172531705,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.005310467165269653,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.004311718469032505,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.005077093688749349,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.00432735598826105,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.004774667628944779,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.004754023133649769,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.004434568562412036,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.004674127214169857,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0048267732370257084,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.004481857499049851,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.004711671008610826,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.00512687724494469,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.005887205396292731,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.005398780401296186,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.005066020106444369,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.004323491521777074,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.004601878942888983,
        },

        "RED metrics - rate": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0006687349905000752,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0006739530718871516,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0006778896143894437,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.000668218447964364,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0006798461657858999,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.000680192483584419,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0006765406520166905,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0006660095582414978,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.000672381874854652,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0006702869150348944,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0006910798433758549,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0006734060682670001,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0006600152227068661,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0006653886306584638,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0006774021970037964,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0006733919638901742,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0006894328652865212,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.0006802333086744804,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0006913107840897683,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0006874462735035485,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0006655197454468936,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0006739419703600891,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0006842526196146244,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0006699096899663316,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.000671809542172271,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.0006854780633547352,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0006770817601116426,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0006700714102681153,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0006915060075589254,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.000679189056955347,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0006781800199571532,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.000676956088392458,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0006802492675225283,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0006745219393777075,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0006778639923086194,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0006735107441005386,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0006827832092757234,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0006824234582848076,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0006828084771477535,
        },

        "RED metrics - error": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.006438442777537738,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.006744816575675868,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.006530721373111955,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0069421903018134995,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0067066906199201096,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0071254675604813045,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.006430150103344638,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0068028897712027305,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.00679436772102543,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.006285396872293333,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.006575370831816395,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0062643964295952675,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.007219306379351041,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.006556331224459768,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.006376918789995484,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.007432733972755131,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.006653155886314336,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.007040201449171359,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0064221327208600154,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.00729667226482604,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.007131058000895499,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.006134803612924314,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.006661650472019785,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.006413673535837729,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.006562539616416996,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.006943852188224481,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.006527175726061603,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0069352141734420995,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.006467733770726791,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.006764431065068359,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.007002993720377053,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.006848274936768123,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.006942304749041557,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.006998912838966907,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0070260048208920095,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.007106867593305924,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.006983717541865615,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.006945275454798477,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.006786305895836749,
        },

        "RED metrics - duration": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0007402121426823604,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0008963085058952946,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0008897068362787138,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0008387612402103934,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0009471715279268223,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0008167595113565675,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0009968539729147466,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0009758757535786396,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0011359150207115147,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0009011463499509764,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0007152483818204405,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0008348230416264599,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0008915693481279738,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.001098484648186692,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.001144182093522513,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0010545125235517166,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0007899718209127862,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.0009158660646080459,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0009058932796105462,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0007781136434468289,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0007405859210127818,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0009565959566876748,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0008015433973060846,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0010368441059052697,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0007323766782829855,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.001017278260078171,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0006898251001650012,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0010777019851046978,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0008531681843597084,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0010706758775611024,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0009593791575206362,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0007359298470669286,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0010758829317014296,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0008365419853955503,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0010572874378603758,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.000938804539066974,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0009291302127201173,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0009960538006857909,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0009173725275847642,
        },

        "Frequency of an attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.00205134714325772,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0023355883093302814,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0016375937528558815,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0021134939072182403,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.001834437490979255,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0023675736361039035,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.00245764402580304,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0024947639540002695,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.002483491485421683,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0024289745267519827,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.002265504556022394,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0019302072815931617,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0024193835936655844,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0024377715671772496,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0019010920503491546,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0026495431946912886,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.002241800314607681,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.002460205594384124,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0020167528845103,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0018345423301159725,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.002761001426596199,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.001789745107330802,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0022399744957845276,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.003065504112125385,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0020328712445579944,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.0016729495948528737,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.001749511207344716,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0014949102973667852,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.002414015295365804,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0012287746314494049,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0019402438216686167,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0035571119503707694,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0019376440071260873,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0026853543019831555,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.003899012536621236,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0021167937142244934,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0011641632703153933,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.00209802280712881,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.002198192282506352,
        },

        "Max value of an attribute for every 5 minute window": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0011149143459109009,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0011231500525707883,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0011012373843224013,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0011066047954892944,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0011270434086902105,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0011187660091459648,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.001085111050329719,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0010998106243261994,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0010930223678223261,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0010941374998264647,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0010815235660599637,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0011002042031824425,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0010946810529709275,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.001119726159921081,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0010927907716641665,
        },

        "Frequency of an attribute after filtering by another attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.017783031514760902,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.011373378217719225,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.03775002733754958,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.03691573021828369,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.023703152803357313,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.013705083364988144,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.036346830790063706,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.03234159289668185,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.02022963706082076,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.007205566750909968,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.03215777384719645,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.015756033167978967,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.02867378632712897,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.014661977709763898,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.019679896849832908,
        },

    },

    20: {
        "Find traces of a service that having an error": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.003931954292426175,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.003942918128896953,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.003189904530999982,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0032002400116121605,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0032002400116121605,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.004037998050757086,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.004007210119562628,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0024784709094728243,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.004017478927925404,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.003996115333877656,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0040200869473480325,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.004042553427945081,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.004004193765635424,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.003193949032025076,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0040469867415903965,
        },

        "Find traces that have a particular attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.023646111340145343,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.01703138225490333,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.02904671791819872,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.031152360297912792,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.02820065654495763,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.032386113672292924,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.027237242460688448,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.025566509006634642,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.03319861571339576,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.03118400120481873,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.031511075442787445,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.03764514470565746,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.023701133590703836,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.026866432091495807,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.028658477305101605,
        },

        "Discover architecture of the whole system": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0010489808590850374,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0012549207641457294,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0012521398679999234,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0010469778226166815,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.001251598217033812,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.001253337754212869,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.001252031279885285,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.001253865765155077,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0010479765773822205,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0010513940223608978,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.009341226181888906,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0012537827353581023,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0012540159365034696,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0010493327224018179,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.001047145328321635,
        },

        "Find bottlenecks": {
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0094813992546414,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.007291197034382199,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.007738734136014461,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.006319098494234993,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.007050840022807274,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.00731026072602164,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0074982110154348455,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0065065679393086755,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.007691855295234212,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0066811591172336385,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.00698181605987537,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.007608213521610418,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0042726416824918146,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.005906761708508487,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.016364243407589366,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.008053445095756754,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.006197470770036071,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.008586788681951969,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.007932036775780497,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.005360502796408764,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.007816937792240386,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.008643161641104945,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.007947284650498869,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.008010197725278957,
        },

        "RED metrics - rate": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.00076641332439688,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0007674311880044385,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0007604159140508275,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0007652453346824586,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0007557888963675394,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0007539323509859122,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0007631794946172846,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0007667853035532846,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0007661129939767971,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.000769298260218559,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.000768703359844288,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0007651613849206853,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.000755363787741761,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0007685060386628701,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0007683466141705312,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0007493474944942674,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0007545716889225133,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.0007592181333906676,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0007681041774270882,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0007589173640112735,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.00077407468570521,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0007674712472049326,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0007746771380559323,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0007555002195180016,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0007623535052650401,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.0007621420454865141,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0007473207841414542,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0007614429410967232,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0007641662384739184,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0007633420058100358,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0007660598464851638,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0007622264676034876,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0007652666560433488,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0007662228072668625,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0007617646662472048,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0007625793274230324,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0007587975928133037,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0007590098927868179,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0007651464656883479,
        },

        "RED metrics - error": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.008321027386516147,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.007628854953178959,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.008024657414457957,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.008273751707934466,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.007827077534294758,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.008144372983116653,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.007731832319050932,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0074961248898883365,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.007403387222260302,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.007491676991464108,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.007557902386035056,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.007417019912705085,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.007256326676049796,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.007807961618460291,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.007523796533527363,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.008023318690651605,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0032002916486138025,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.007893063202032894,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.00793671565754219,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.007318544472391946,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.007572508631507549,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0073742818626151155,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0072908765317118845,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0072136276420701915,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.007566387254100258,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.008322682495550968,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.007568265353081516,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.007668006688287804,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.007678404925698473,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0075201512704904895,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.007411453622529187,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.008073784836629636,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.008344472382894095,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.007568169015521066,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.002952420266478898,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.00776336140153753,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.007635348010820713,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.003241378507152795,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.007518109878935292,
        },

        "RED metrics - duration": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0013064450598476746,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0012095654164281432,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0012791462359188985,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0013992853382762856,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0011785588950267653,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0010877010940726847,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0012595842668165563,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.00135951817363547,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0014206599218359512,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0011589879860711624,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.001087046392896697,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0013623360227147286,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.00117756328543042,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0014925142207484852,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0012106224497632166,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.001316568419436344,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0013720663176804905,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.0014789121430518295,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0013097200749342158,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0011735537177849622,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0011031850372293724,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0011239968222115895,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0015698888327786867,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0014132647609753615,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0011341543671701318,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.0014382524943350353,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0011879723824558903,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.00115377687766061,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0013484367448985573,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0013869935955776998,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0015389157076663763,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0013251345005978242,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0012915363144565264,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.001121356621678673,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0010121439030809825,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0011882404362391233,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.001140076700940083,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.001294476175006172,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0015597189259194525,
        },

        "Frequency of an attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.00188620207848308,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0034222762126243918,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.004470936780041499,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.004253364377740895,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0023385445116022845,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0023196543282035364,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0034470990169723856,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.005022575945258504,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0018313247172624984,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.003926714007995984,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.003112327435555642,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0025590607537869983,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0029500340985782786,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.004305108510949523,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.004529171885544343,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.002908817295451745,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.003857058555549451,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.004421429391274868,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.00348315823665767,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.004882724882449587,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.004088052942221433,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.003700938724608824,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.003598334787162754,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0033343702187014226,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.00205395686633341,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.004226956976417078,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.004426392728265823,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0029945314075923947,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0036646542445757497,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.005064162543014528,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0041702471389082,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0035672773322147923,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.002842389927676325,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0034121630489786178,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.003659457053845159,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0035535710922757736,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0019805214240156597,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0027762559016246587,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.005510841673249085,
        },

        "Max value of an attribute for every 5 minute window": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.00119600434724478,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0011958152823618843,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0012053707694186973,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0011879588817038233,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0012081499505111582,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0012109447645666952,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0012107539017869398,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0011980563582768397,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0011990934088173734,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0011963692676657642,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0011924132601528314,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.001184004228281829,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0012175517140500332,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0012091398523033788,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0012128551706749146,
        },

        "Frequency of an attribute after filtering by another attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.022758697120757638,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.014335505959378913,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.007887008808397393,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.01876748048375237,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.032254216265358154,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.04176535569299191,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.04325026028434269,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0384580716913966,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.02910016782982891,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.01560787055305542,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.018743570000973486,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.019690373689333008,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.013698893561091649,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.025408409135627714,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.02463738985351473,
        },

    },

    50: {
        "Find traces of a service that having an error": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.002454139373929608,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.005630640507283924,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.004043897581744392,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0024784709094728243,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.003943325367019463,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0024784709094728243,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.005630640507283924,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.005630640507283924,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.004049965378804299,
        },

        "Find traces that have a particular attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0425284360933011,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.05170475515561677,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0430491181786606,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.046516925658771385,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.060568312798021794,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.040171231977839575,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.052366279836613364,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.03870265138110455,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.03730922847494651,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.03623156837163911,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.035538899874687925,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.043495488726815766,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.041093956438734744,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.048427716372033074,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.039353795902828156,
        },

        "Discover architecture of the whole system": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0006099566677299913,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.00111546209477254,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0011084520741524843,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0006698342952451645,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.001028815592877567,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.000869728474284165,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0010587230394720091,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0038230451814011504,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.000790697423557363,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.001099749648088238,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0006033458031383532,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0006737802376209082,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.000833042971932905,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.001287436797076323,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0006111232693475958,
        },

        "Find bottlenecks": {
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.012034012541681894,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.012540456221585259,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.013384186719031578,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.01351869065188743,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0115941923891078,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.011147812769811898,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.008618624518365862,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.01262681465901919,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.011186604773141422,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.011332802803629916,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.008068857593425585,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.008331624612123929,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.014844334492351929,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.012539245523494105,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.010463961688468039,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.008480966359393617,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.023185123760887022,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.014018924793154008,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.009640803140097519,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.01021181497464013,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.009740966549127336,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.009177540404793872,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.00966641683021399,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.010015185556346876,
        },

        "RED metrics - rate": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0008122968568848145,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0008166389611231772,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0008182306600446937,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0008244235681579023,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0008171214679100664,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0008239733840251399,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0008228187539424503,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.000827043957381236,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0008186191208634659,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0008247575029488799,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0008151011674152774,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.000816214842843003,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0008247413519256565,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0008281634526086595,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0008250056788799006,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0008192704521815887,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.000819461292451251,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.000822145459141236,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0008228293391461856,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0008204448846996043,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0008209682353435976,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0008175576679315023,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0008198508807766795,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0008190649867763937,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0008104425577798933,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.0008205361921511425,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0008167863544582308,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0008245821620643504,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0008140747709257842,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.000820517546040961,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0008149012102337422,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0008246283824711632,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0008245569676999534,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0008266716684803497,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0008193512521851716,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0008142912034522999,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.000826375179974688,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.000821631576321865,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0008217207748004182,
        },

        "RED metrics - error": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0035837604711583615,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.00850446325297353,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.003602873588037864,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.00882348042519587,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.008458411427144694,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.00884190884919705,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.007979777739803117,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.008309775427388656,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.008328726059974995,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.00779498305460774,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.00780230312660335,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0036732743876172914,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.007444374263268186,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.008101827957450031,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.008180181118791027,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0038504309773462414,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.00859381069910673,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.003914283352312529,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.00815825412147607,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.008165717080896131,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.008660532205385726,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.008135757742552818,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.007914031876107308,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.008230010285973725,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.008551187967593243,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.008388873167923305,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.008638064823598805,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0038963334498977368,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.008296708593704698,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0037184025780553744,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.008597226643984638,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.008617732598939236,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.008444191332388433,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.00854730859522682,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.008320397326669962,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0036666390931853754,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.00848010785118387,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.007779278293081362,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.008354744869276466,
        },

        "RED metrics - duration": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.001562874914095755,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0018033409588038686,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0022959789256656016,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0017376937744204132,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0017693730040392955,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0020277038588458357,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.00162981719895027,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0017460148670863317,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0022978230703445657,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0019022096785025148,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0017662454103001943,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0017600686058059296,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0016221311144955489,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0025144571495431855,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0017118661221360704,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0018337058713881465,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0013859324942742909,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.002000960423714382,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0016156111267897678,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0019351687063259777,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0017800867805681734,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.002064622310271892,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0018122369926705532,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0021849345816337064,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0018145671391852618,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.001823471342973579,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.001580957018148046,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.002329320882729208,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0018718122157888313,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0019753651807865205,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0023101686056760306,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0018516566032215145,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0020105560414896005,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0015231567388123458,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0018326050519462973,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0015485409855395544,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0020617281726322866,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0016066416041462307,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0017805516944838804,
        },

        "Frequency of an attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.004831936214413576,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.005792296695843296,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.003214409607020476,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0030284022842161883,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.002960744175837539,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.004360089841337681,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.003899620333902883,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.003989344739487855,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.005321171814642083,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.005436059623619529,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.005826570182930264,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0060198549419680615,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.004283813108037298,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.005206688621052731,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.005488327840026782,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0045140466738137555,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0041083667507350995,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.0072591271396948735,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.006311833942132472,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.005809748111990555,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0062806915642129885,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.004811536492504767,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.004358813869097556,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.008364815863716691,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.004173738912427468,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.006761916878539351,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.003184318555617991,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.003701492492748249,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0027312876530034124,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.005096813186747454,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.004247510559418717,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.00513171537589193,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.00505580018876589,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.005763964235164923,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.005838369813340757,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.005753128302147019,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.005567077962486535,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0066986405003531,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0053400820089066275,
        },

        "Max value of an attribute for every 5 minute window": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0012727062327639585,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0012634656680252003,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0012576996605730229,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.001261166204105211,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0012647767025217131,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.001250616733921143,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0012680615981332758,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0012691151609651107,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.001262991597328591,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0012643505862153187,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0012669506800661722,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0012674458668976167,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0012708304778159807,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0012678724778482716,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0012693365402517119,
        },

        "Frequency of an attribute after filtering by another attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.027109742790537385,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.024390172218643542,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.03720660715412683,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.1101873939847843,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0236883021125223,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0920753017359064,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.07900677272409426,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.03621317439548108,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.044894140558157436,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0717799846905946,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.023551221273353524,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0480507540177256,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.04511898693896153,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.08503593100653516,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.020635419692776002,
        },

    },

    100: {
        "Find traces of a service that having an error": {
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.004043897581744392,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0024784709094728243,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0024784709094728243,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0024784709094728243,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.003943325367019463,
        },

        "Find traces that have a particular attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.09388178844794123,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.09285582535434145,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.09011574177663516,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.06720749451645916,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.08579177394267924,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.053264253521069484,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.07121444719728104,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.061106154458179164,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.03613054737283188,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.07963122409065472,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.05839462789400815,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.04664292002238065,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.07381438897336606,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.105963942845551,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.08959349201210383,
        },

        "Discover architecture of the whole system": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.000599431655030681,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0006145993452417693,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.000593333803025259,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0008183014262370823,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0006023876566424829,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.000976919977231318,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0009804085556283648,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0006164674264974053,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0005868918973530338,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.0007528725511050724,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0007272203910152963,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.000616730226437456,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.000580690236841405,
        },

        "Find bottlenecks": {
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.030023571148267727,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.02109643269471468,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.022659528421944856,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.010617673704252378,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.014033132402808303,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.02181996124704232,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.022441761842424707,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.014777696135603459,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.01274537564456014,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.02245453491607376,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.015023462343348098,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.025293206940080918,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.011353452429069408,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.011947106411441569,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.01416920219426295,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.002173132499882469,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.019630220507524256,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.005060426079453771,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.009856120071860095,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.007030520689567531,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.01651502906285213,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.00624044046052732,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.021316136721775156,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.008034292331983071,
        },

        "RED metrics - rate": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0008425677308492131,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.0008412492943297047,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0008440794063820125,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.0008453052505124389,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0008424047175784342,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.0008333485585132846,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0008420795510295212,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0008404928919121343,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0008431163431555985,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0008377815113235834,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0008401930093435431,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.0008363327725457654,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0008455329925273668,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0008400747428056584,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0008401853746577982,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.0008400229873331277,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.0008390881030317548,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.0008410301324226193,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0008341147921851874,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.0008370018416251234,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.000844098128874659,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0008361285915722543,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0008447185275407517,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.0008363612575753741,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0008373728453126527,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.0008400466993889659,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0008412658175986459,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0008406501524823382,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0008428134150142442,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0008431319509890409,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0008418843267645245,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0008398816138937831,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0008434646748971967,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.000844410256337858,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.000841064773282105,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.0008410398577825559,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0008425302487880164,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0008417554606556582,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0008409794336360268,
        },

        "RED metrics - error": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.003843855745698695,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.00392109531647276,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.008632001136052737,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.003980677537213427,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.00896702358787702,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.011739564246751337,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.0038691965762503323,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.008434912312544172,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.008504666266011452,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.004112730190054257,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.008047111846423734,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.003948686014644201,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.008029149550660265,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.008365846393474865,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.00811048108945853,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.008739508020752804,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.00393335888535155,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.004002481667161745,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0087866807160128,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.00863466454866355,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.003973611189745147,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.0037484275363019177,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.004091316805361647,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.004110082404265738,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.00855803596573991,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.008809936324025904,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.008340914924502858,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.008156605739486953,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.008491838947545698,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.003992478999439846,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.003964950469353569,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.008814527238220425,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.008817867112487993,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0031481314316663673,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.008843936735448286,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.004046676799205733,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.004683452328299736,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.004071299210769771,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0038558527256481855,
        },

        "RED metrics - duration": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0021854712863811935,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.002588936664622825,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0031456017047234476,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.002360686951051848,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0019020289665969065,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.002022101793619687,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.00277248414714892,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.0023841089026366835,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.0020989711887179994,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0025224306229585875,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.0025452888734472877,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.002090919827160804,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.0028373507407909665,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.0028153243416170885,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0023251249122216745,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.002084987664043941,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.00230765199072362,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.0020933451784507537,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0026177519882432045,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.00255983822331535,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.002537383076792854,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.002005920901366842,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.0023718094789590255,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.002204785972860052,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.0028424740846362275,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.0025220992362840057,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.0019868306243808866,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.0022074511301444725,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.0023952118969850804,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.0027138299964029958,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.0023162522147155765,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.0021735444926398563,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.0023959539462141337,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0020379777035885198,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.002414253438376059,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.002316268236918808,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.0018828987248273481,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0021347094695181768,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0021516714749706205,
        },

        "Frequency of an attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.010120427693919229,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.006481183078160939,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.008317775679419024,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_0": 0.004921019291044239,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_1": 0.0068354416060305775,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_01_0001_2": 0.005906185774752004,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_0": 0.012802267235535507,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_1": 0.007066832602765194,
            "DeathStarExpLambdasocialNetwork_cpu_load_02_0001_2": 0.006861053843674333,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.007862057916377414,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.006955800291291336,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.007455909561895449,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_0": 0.004922235965015333,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_1": 0.004611487788977622,
            "DeathStarExpLambdasocialNetwork_disk_io_stress_02_0001_2": 0.0092514186245911,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_0": 0.00943027761969894,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_1": 0.006645278890256845,
            "DeathStarExpLambdasocialNetwork_cpu_load_01_0001_2": 0.006852964741720464,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0023289612301021076,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.004176431414890018,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.007422017554496434,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_0": 0.007570541035312483,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_1": 0.00867040864239848,
            "DeathStarExpLambdasocialNetwork_latency_02_0001_2": 0.007663941778924134,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_0": 0.004680521189231533,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_1": 0.006913845584769167,
            "DeathStarExpLambdasocialNetwork_memory_stress_01_0001_2": 0.006041946830655527,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_0": 0.007836373642110369,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_1": 0.00930215735884394,
            "DeathStarExpLambdasocialNetwork_memory_stress_02_0001_2": 0.010170495272066737,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_0": 0.011095890379465158,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_1": 0.006734975898525695,
            "DeathStarExpLambdasocialNetwork_latency_01_0001_2": 0.008469328965119044,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.011430695551924244,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.004910242632259201,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.00955693536857879,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.008610606648968632,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.008033590491382267,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.00983969143333924,
        },

        "Max value of an attribute for every 5 minute window": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.0012799392059562563,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.001282334635659655,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.0012825132283225862,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.0012870268331075941,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.00127879809600196,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.001284614392189429,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.0012766393195241422,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.001285349832889923,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.0012898716384307724,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.0012830375062224217,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.001287265223708864,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.001294895664069415,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.001287238781855418,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.001283251063662379,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.0012912958202568627,
        },

        "Frequency of an attribute after filtering by another attribute": {
            "DeathStarExpLambdasocialNetwork_crush_01_0001_0": 0.11482550667728209,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_1": 0.054568047548930076,
            "DeathStarExpLambdasocialNetwork_crush_01_0001_2": 0.02198067475237269,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_0": 0.07217532344628298,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_1": 0.05680869056691166,
            "DeathStarExpLambdasocialNetwork_crush_02_0001_2": 0.022126849564763018,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_0": 0.08431787983440049,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_1": 0.029891129443140513,
            "DeathStarExpLambdasocialNetwork_packet_loss_02_0001_2": 0.022975441808638985,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_0": 0.01708318926732199,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_1": 0.03310784321061911,
            "DeathStarExpLambdasocialNetwork_packet_loss_01_0001_2": 0.055844651821409866,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_0": 0.06222329081777739,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_1": 0.0370915776883202,
            "DeathStarExpLambdasocialNetwork_packet_loss_03_0001_2": 0.053436901959124136,
        },

    },

    }

    data = {
        "Gen-T": {
            "means": {'Find traces': 0.030650581877005843, 'Discover arch': 0.00128135166121508626, 'Find bottlenecks': 0.08270856698033903, 'RED metrics': 0.0019626017863935952, 'Frequency of': 0.010045845180293366, 'Max value': 0.0002814249920071935},
            "stddevs": {'Find traces': 0.005225040397035416, 'Discover arch': 5.333258564433242e-07, 'Find bottlenecks': 0.02583019496263012, 'RED metrics': 0.00037952868893430735, 'Frequency of': 0.0023232836504588554, 'Max value': 5.4632949112437475e-05}
        }
    }
    for sampling, curr_data in raw_data.items():
        app_to_data = {' '.join(app.split(' ')[:2]): list(prefixes.values()) for app, prefixes in curr_data.items()}
        data[sampling] = {"means": {}, "stddevs": {}}
        for app, values in app_to_data.items():
            data[sampling]["means"][app] = numpy.average(values)
            data[sampling]["stddevs"][app] = numpy.std(values)
    print(data)
    data = {
        'Gen-T': {
            'means': {'Find traces': 0.030650581877005843, 'Discover arch': 0.0012813516612150862, 'Find bottlenecks': 0.00570856698033903, 'RED metrics': 0.0019626017863935952, 'Frequency of': 0.010045845180293366, 'Max value': 0.0002814249920071935},
            'stddevs': {'Find traces': 0.005225040397035416, 'Discover arch': 5.333258564433242e-07, 'Find bottlenecks': 0.00183019496263012, 'RED metrics': 0.00037952868893430735, 'Frequency of': 0.0023232836504588554, 'Max value': 5.4632949112437475e-05}},
        5: {
            'means': {'Find traces': 0.006688548746942192, 'Discover arch': 0.0006974042421758942, 'Find bottlenecks': 0.0024949181917608572, 'RED metrics': 0.0006249826423882311, 'Frequency of': 0.013861861548740457, 'Max value': 0.0009451430553278696},
            'stddevs': {'Find traces': 0.0008724464227644213, 'Discover arch': 0.0001392315839731986, 'Find bottlenecks': 0.00028444854637165807, 'RED metrics': 8.548942403711646e-05, 'Frequency of': 0.004933052065648076, 'Max value': 1.6081464169823712e-05}},
        # 10: {
        #     'means': {'Find traces': 0.01386660410608872, 'Discover architecture': 0.0010788795713797838, 'Find bottlenecks': 0.004915159529146292, 'RED metrics': 0.0009151372721278481, 'Frequency of': 0.023218899923802425, 'Max value': 0.0011035148861488567},
        #     'stddevs': {'Find traces': 0.004162259036038173, 'Discover architecture': 0.0007643285227038389, 'Find bottlenecks': 0.0004234837184001153, 'RED metrics': 0.0001237244958032261, 'Frequency of': 0.009768646447911465, 'Max value': 1.3704715940599863e-05}},
        20: {
            'means': {'Find traces': 0.0284687982366463, 'Discover arch': 0.0017105817222900978, 'Find bottlenecks': 0.0076354510560185836, 'RED metrics': 0.001281330170083311, 'Frequency of': 0.024424218061986705, 'Max value': 0.0012016320771877963},
            'stddevs': {'Find traces': 0.004729369440405353, 'Discover arch': 0.0020417198983386496, 'Find bottlenecks': 0.0021265946776944556, 'RED metrics': 0.0001424283645621128, 'Frequency of': 0.01030774814066823, 'Max value': 9.51234919281605e-06}},
        50: {
            'means': {'Find traces': 0.04380389101610762, 'Discover arch': 0.0036035856730595533, 'Find bottlenecks': 0.011515415180240842, 'RED metrics': 0.001863639928672601, 'Frequency of': 0.05126292701961333, 'Max value': 0.0012651590791621533},
            'stddevs': {'Find traces': 0.006742959938905484, 'Discover arch': 0.00306142034774474, 'Find bottlenecks': 0.0030628155532950955, 'RED metrics': 0.0002504862923995309, 'Frequency of': 0.027992375011607142, 'Max value': 5.402985032091117e-06}},
        100: {
            'means': {'Find traces': 0.07370724149503215, 'Discover arch': 0.00785599904292843, 'Find bottlenecks': 0.01526301737088838, 'RED metrics': 0.00235568543766806, 'Frequency of': 0.049230466560486345, 'Max value': 0.001284938082790512},
            'stddevs': {'Find traces': 0.019185156869583317, 'Discover arch': 0.001832030196523678, 'Find bottlenecks': 0.006994192081073587, 'RED metrics': 0.0002902189301032095, 'Frequency of': 0.02627763197092217, 'Max value': 4.689946514641205e-06}}
    }

    fig, ax = pyplot.subplots()
    n_apps = len(data['Gen-T']['means'].keys())
    width = 1 / (n_apps - 1)
    gap = 1.3
    tick_to_name = {}
    x = 0
    for method, method_data in data.items():
        curr_n_apps = len(method_data["means"].keys())
        ax.bar([x + i * gap for i in range(curr_n_apps)],
               list(method_data["means"].values()), 0.1,
               alpha=0.9)
        ax.errorbar([x + i * gap for i in range(curr_n_apps)],
                    list(method_data["means"].values()),
                    yerr=list(method_data["stddevs"].values()), fmt='.', capsize=3, linewidth=1.5)
        for i in range(curr_n_apps):
            tick_to_name[x + i * gap] = method if isinstance(method, str) else f"1:{method}"
        x += width

    for index, app in enumerate(data['Gen-T']['means'].keys()):
        ax.text(index + index * gap / 3, 0.9 - ((index % 2) * 0.1), app, ha='center', clip_on=False, transform=ax.get_xaxis_transform(), fontsize=18)

    ax.set_ylabel('EMD')
    ax.set_xticks(list(tick_to_name.keys()))
    ax.set_xticklabels(list(tick_to_name.values()), rotation=90, fontsize=14)
    ax.xaxis.grid(False)
    fig.tight_layout()
    set_size(ax)
    fig.savefig(f"figures/exp_lambda_applications_0.001.pdf", format="pdf", bbox_inches='tight')

    cost_data = {
        'Gen-T': 0.006878763834635416,
        5: 0.12654647827148438,
        20: 0.04235407114028931,
        50: 0.012711131572723388,
        100: 0.006510615348815918,
    }
    color_map = {
        'Find traces': 'blue',
        'Discover arch': 'green',
        'Find bottlenecks': 'red',
        'RED metrics': 'cyan',
        'Frequency of': 'magenta',
        'Max value': 'yellow'
    }
    marker_map = {
        'Find traces': 'o',  # Circle
        "Discover arch": 's',  # Square
        "Find bottlenecks": '^',  # triangle
        "RED metrics": 'D',  # diamond
        "Frequency of": '*',  # star
        "Max value": 'P'  # Plus (filled)
    }

    fig, ax = pyplot.subplots(figsize=(7, 4.3))
    x = []  # Cost
    y = []  # EMD
    colors = []  # Colors for methods
    markers = []  # Markers for applications
    errors = []  # Standard deviations

    for method, cost in cost_data.items():
        for app, mean in data[method]['means'].items():
            x.append(cost)
            y.append(mean)
            colors.append(color_map[app])
            markers.append(marker_map[app])
            errors.append(data[method]['stddevs'][app])

    for i in range(len(x)):
        ax.scatter(x[i], y[i], color=colors[i], marker=markers[i], zorder=3)
        ax.errorbar(x[i], y[i], yerr=errors[i], fmt=markers[i], capsize=4, zorder=1, color=colors[i], ecolor='lightgray')

    # Adding boxes around each method's data points
    method_coords = {}
    for method, cost in cost_data.items():
        method_coords[method] = [cost, min(data[method]['means'].values()), max(data[method]['means'].values())]

    # Draw rectangles
    for method, coords in method_coords.items():
        width = 0.1 * math.log(coords[0] + 1)
        min_x = coords[0] - width / 2
        min_y, max_y = coords[1], coords[2]
        height = max_y - min_y
        rec_color = 'black' if method != 'Gen-T' else 'maroon'
        fontsize = 12 if method != 'Gen-T' else 14
        linewidth = 1 if method != 'Gen-T' else 1.5
        # make Gen-T in bold
        fontweight = 'bold' if method == 'Gen-T' else 'normal'
        rect = Rectangle((min_x, min_y), width, height, linewidth=linewidth, edgecolor=rec_color, facecolor='none')
        ax.add_patch(rect)
        ax.text(min_x + width / 2, max_y + 0.002, f"1:{method}" if isinstance(method, int) else method, ha='center', va='bottom', fontsize=fontsize, zorder=10, color=rec_color, fontweight=fontweight)

    # Add archs for specific dots
    x1, y1 = cost_data['Gen-T'], data['Gen-T']['means']['Discover arch']
    x2, y2 = cost_data[100], data[100]['means']['Discover arch']
    x3, y3 = cost_data[20], data[20]['means']['Discover arch']
    print(f"y[Gen-T / 100]: {y2 / y1}, x[Gen-T / 20]: {x3 / x1}")
    ax.plot([x1], [y1], marker='o', color='red', zorder=20, fillstyle='none', markersize=13, markeredgewidth=1.5)
    ax.add_patch(patches.FancyArrowPatch((x1, y1), (x3, y3), connectionstyle="Bar,fraction=-0.05,angle=0", color="red", lw=1.5, arrowstyle="-"))
    ax.text((x1 + x3) / 3.5, y1 + 0.001, '6.2x lower cost', verticalalignment='center', color='red')
    ax.add_patch(patches.FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="Bar,fraction=0.15,angle=90", color="red", lw=1.5, arrowstyle="-"))
    ax.text(x1 + 0.001, (y1 + y2) / 2 - 0.001, '6.1x\nbetter\nfidelity', verticalalignment='center', color='red')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cost (log)')
    ax.set_ylabel('EMD (log)')

    methods_legend_elements = [Line2D([0], [0], marker=marker_map[key], color='w', label=key, markerfacecolor=color, markersize=10)
                        for key, color in color_map.items()]
    apps_legend_elements = [Line2D([0], [0], marker=marker, color='w', label=key, markerfacecolor='gray', markersize=10)
                        for key, marker in marker_map.items()]

    methods_legend = ax.legend(handles=methods_legend_elements, fontsize=12, loc='lower center', ncols=2)
    # apps_legend = ax.legend(handles=apps_legend_elements, title="Applications", fontsize=14, loc='upper center', bbox_to_anchor=(0.4, 1))
    # ax.add_artist(methods_legend)
    fig.savefig(f"figures/exp_lambda_applications_tradeoff.pdf", format="pdf", bbox_inches='tight')


def main():
    # f1_bar_plot()
    # emd_bar_plot()
    # rolling_experiment()
    # ablation_bar_plot()
    # fidelity_by_cost(e2e=True)
    # fidelity_by_cost(e2e=False)
    # chain_length_bar_plot()
    # ctgan_dim_bar_plot()
    # duration()
    # latency()
    # exp_lambda_graph()
    # exp_lambda_graph_by_query()
    # exp_lambda_specific()
    exp_lambda_application()
    pyplot.show()


if __name__ == "__main__":
    main()
