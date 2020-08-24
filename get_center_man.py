
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import csv

num_list=['15807']
# result_list=[]
# need_to_fix = [22604,38478,23962,37092,22405,22738,19036,19454,32920,32568,23371,11634,21513,10970,20496,11107,26775,11627,17819,20087,16352,22450,11858,28660,15254,15742,34006,36490,20079,11959,39823,35132,13395,35547,15594,34517,23467,30695,24838,14208,22376,26034,30637,17058,32146,16314,23760,11899,18844,33853,37831,29187,18111,28128,32533,25330,34820,38009,33942,27145,22900,32353,17518,12670,37370,24157,24923,21304,34625,24397,36197,11319,39933,21328,37231,16350,16683,19896,25060,31822,34903,21670,32742,18324,30655,28968,32372,33892,21954,11754,25855,29626,19287,30029,26041,18758,24367,23094,11650,24514,38518,20599,17348,37541,10205,32266,16332,37035,12626,21074,21189,15833,32670,27740,20497,17317,17269,15286,31640,18911,16380,21129,27971,14422,23615,11822,23074,15057,12000,39713,16692,33441,20233,15452,38076,28991,38422,38867,37356,13037,24631,30000,36308,27557,27499,21977,18591,21266,23810,27075,10946,36880,38851,27054,26955,18257,34715,28300,38333,13320,15010,18492,30713,25769,34713,24220,16936,39551,15030,33225,18702,10010,30423,31915,13159,39607,29127,30343,20321,15964,37149,39973,20757,10327,27751,19562,21105,23485,15398,36199,13518,11800,19374,33267,25071,20420,18057,20194,33928,38849,35838,11037,26223,29351,14131,29902,12884,37835,29723,35379,22628,18824,22836,20605,26217,36259,26302,15122,32368,12522,13802,21509,27425,10154,16987,16961,15227,35888,13790,39480,38958,31754,37674,38232,33715,29678,21965,11240,21824,39548,31917,19955,29062,18623,31911,37357,23386,13536,17223,27770,38306,19858,10344,16920,17250,26336,12989,20352,29293,39690,17501,11696,28748,12729,39389,29815,27859,33086,24785,32822,28762,37114,29993,19139,31312,25770,21272,26588,14676,32760,19737,19249,24268,16699,28231,18062,17791,12702,15259,26851,30591,28591,27826,35033,28021,28243,23261,37815,25979,16516,39388,16555,23438,27633,14522,34271,15880,34425,36730,27515,25042,33583,21787,17634,11195,20384,22174,32617,18357,34646,32221,33542,13440,32770,10617,15315,29088,37379,16776,10832,21839,18589,17027,37567,21530,15260,32623,32072,28960,25139,22906,20426,37781,18272,17321,15608,16434,23084,17648,15426,16295,13563,29018,21351,32247,24763,22922,35668,11594,23497,27847,24333,12026,20721,14853,11469,39771,27296,13308,12848,18651,37390,18865,22785,37153,23500,18712,30219,34705,12572,39787,38177,31631,22579,18239,34194,38969,14737,10213,20839,36595,35661,30646,13621,37534,29294,18432,33346,24198,36476,10904,12502,23969,27473,12184,22020,28825,32663,23744,16712,15885,27774,33685,36208,12123,23896,19630,16511,28258,25969,20212,32101,37029,39362,17526,32904,20578,37467,17810,29230,15317,21234,30576,15088,14874,26472,34091,33005,16789,22885,11793,24354,10665,34913,16290,17508,20008,23151,11146,13104,21086,34217,23927,13200,35715,24798,19880,36804,19434,23580,22327,25986,31705,31618,12209,20961,18330,35765,18422,28510,36802,32924,30278,35717,22205,26823,18786,17479,13725,28115,10551,11680,39930,11288,19710,19235,33553,34812,37446,26079,39627,11033,39742,15450,34628,33865,31145,17881,38429,14698,22501,30864,24630,21809,18284,34747,16539,39462,37135,28916,18385,13946,29670,34570,25822,25735,10371,20190,26752,26224,17152,27495,19512,19930,23836,17685,31046,25222,25862,36798,13383,39364,27363,32773,13005,15502,24800,30209,23647,31232,12953,16721,38484,32676,37383,33095,29099,28288,29505,21992,30755,22737,34312,28664,30728,15164,24078,25159,28066,20027,15588,21495,36324,28770,18938,13855,34171,27281,38910,14757,21280,14579,17929,38935,37521,33312,17576,18601,35792,33073,14970,35601,27924,25212,25091,31959,29719,27175,30130,27022,12960,30109,32363,29967,20360,30217,30283,32199,17401,26259,23437,26629,33124,32140,33723,10135,11776,36566,31285,15378,38490,34360,21735,17510,31783,14816,12886,26965,22645,26805,31685,27987,38261,39361,22646,38414,12736,29142,16716,11005,28096,23837,31688,15938,12234,22295,22795,26719,30468,22800,23894,13917,15566,22441,18258,20117,23026,28562,23044,19997,21205,13739,11821,18328,12313,12206,30198,23989,36901,16902,12719,15977,19646,15114,27355,10472,31983,14336,18434,15751,10777,17611,29141,39367,33180,14077,35362,38442,17014,35585,27309,19368,34236,19402,19389,20160,24584,10106,10836,34022,29952,25148,27331,38894,39599,32396,27130,30129,17110,12438,21446,36134,21841,19534,18580,32735,34561,21111,28718,37417,38227,21444,36307,19599,22813,38749,14485,10525,16845,36222,21844,35370,31745,11078,36891,30564,14544,38113,39909,23502,39051,15603,20392,18240,33699,24280,26363,21523,33986,23419,16958,14405,31188,37912,27569,17971,13591,31370,36185,14229,35555,10943,32710,23692,25790,28195,28275,34037,34269,13935,38735,19755,19757,35185,11153,30906,26396,12390,36871,36743,35233,33844,27315,12033,13432,10829,26749,33712,33261,27743,22491,32476,19739,24781,15556,30833,17573,25379,13207,32314,26358,22114,35059,38918,13672,28233,33944,17237,12770,36602,32847,24717,14909,37021,36975,29899,26084,10228,35798,35652,16609,29111,25233,22793,10395,24582,15899,31882,10523,14282,20654,11786,36616,22378,22871,13751,20906,12215,19072,17197,11491,10906,32613,28552,39896,36477,24117,26605,23911,35139,14375,16244,18906,34894,18543,28060,35115,25817,13183,25239,27044,28381,10881,13433,12695,31665,18861,35116,21237,39785,38796,33374,20294,12089,14815,11460,10566,16466,35762,18679,22551,22493,36060,31471,24450,22992,24845,24498,27834,12377,17818,19259,26334,31306,10397,10357,38671,10166,25498,19889,37425,33011,29345,17030,26971,28043,33560,18704,39450,28730,31646,14733,28818,20927,32376,39094,20045,11470,31054,21438,12780,29860,39865,24404,35905,11574,11922,36260,10470,10974,15208,33383,22479,34623,35985,28373,35487,25736,19390,19714,38780,26847,27242,15196,15381,36850,12535,30025,29884,22826,25641,27628,15473,36938,35134,17664,38483,14650,30896,27976,32111,36547,15867,14084,11340,20099,10456,29450,37660,32753,32496,24441,37698,29015,20332,34732,20765,38409,39159,26523,30903,23174,33139,13123,11900,29503,15485,20645,19142,26511,22023,35016,23620,27657,38596,33252,10949,39964,36590,18884,23471,21419,11419,15470,31942,30680,32690,39914,15436,31976,25490,35625,35748,32227,33317,10087,34513,27325,26373,37009,26938,16804,30466,15073,33669,19996,23941,28561,14705,11503,36660,14332,16014,29737,15795,31878,10675,36173,12539,31554,28365,29551,12711,10476,38799,13600,14920,27824,36022,13064,32892,26500,16454,25097,23536,11837,25232,26258,29976,19565,20432,36065,28261,34745,11555,33343,32340,29506,13609,28125,31932,15953,31804,16119,12924,23055,26309,25768,35012,11050,13906,21470,29025,29621,30615,28424,28636,30778,34804,28282,21653,39350,22881,17208,15903,25825,12541,36722,10322,38804,10429,32584,22938,26120,22105,27635,39486,13587,20015,19810,24865,16625,37130,13359,26019,35620,30046,30418,39626,17297,19493,31757,33642,18451,14642,38472,12259,34755,31055,23263,13321,37661,18060,34079,23849,34159,24242,30305,27295,13959,13438,25572,28654,24884,25879,38399,37949,26221,22257,28465,23834,39377,38525,27092,27739,36382,17791,33521,37096,28936,27985,10019,24769,18348,10961,20387,39987,37367,24244,18727,10877,18113,32597,11951,33748,24249,21422,38617,24705,24746,11380,20073,32851,21057,39741,33752,31636,39663,17744,21623,23228,21729,27352,29913,17802,19536,22369,27694,33582,16099,17495,24309,26536,26142,27721,35616,18548,30992,23785,26341,39712,36748,30070,33906,12441,34027,14833,22019,13135,21399,18178,14561,21121,27143,32606,34190,31852,16762,11155,38366,36502,29295,34710,35330,19785,22138,16524,19942,39766,38444,12789,36343,16029,34884,31177,20673,22004,21343,37732,35228,37789,30363,10045,35480,34246,10889,29115,37756,14348,23806,21738,31403,20377,30897,19363,12730,33743,21748,33773,14820,29254,21679,20083,32523,13256,31497,17700,21864,16390,34569,39905,35300,20264,30098,32986,19321,25144,31651,24952,36529,31955,19506,28434,25302,30776,21898,35095,14524,16273,22861,28062,20152,25674,14447,30189,28604,26906,36292,26008,19471,23925,24692,13982,23908,28756,28229,19979,38973,36671,19452,19956,33363,15896,13504,17285,30180,24341,23424,17129,16432,30121,19716,32243,11481,10199,15109,33914,18528,30486,24028,19700,36991,13588,12207,31418,11341,34160,38558,22882,26837,37820,35655,23445,20030,38679,20479,20537,37871,19276,21483,38276,35278,31650,17592,35310,36374,21988,14810,25265,16576,18944,10804,37138,35990,28202,34485,23843,28692,39436,39150,27087,24512,23739,10981,18019,10687,15271,38069,39723,15584,16000,20707,18778,20161,23041,22783,16128,25241,23326,10118,22036,30796,29072,32679,25270,15365,36668,13448,22891,19886,13714,25960,35505,20614,36517,32466,34275,27717,37422,28538,24107,22311,10507,36318,30861,20086,36304,11048,33857,12095,23922,24738,32719,10466,29148,37054,33666,37134,20634,11370,33185,11421,24937,33577,34475,18534,37224,31270,12825,30877,14468,35126,18050,34435,18594,30926,38438,35008,31712,38040,15514,27648,12756,23075,19372,20259,34431,39897,38492,21290,29865,10293,34635,22850,27715,19882,28819,16059,10757,36163,34768,17471,31828,36265,33331,24060,19938,13102,35089,21294,35197,25123,38033,19835,38797,28808,33096,14590,33458,29260,16210,25778,18669,10605,31500,29758,27427,25491,38489,14008,16033,34158,39155,28105,26831,24139,19732,15772,35970,13458,10252,11426,18107,16149,29600,13773,26447,15457,20868,27401,35731,34039,23285,13263,37695,35777,20171,21766,28719,12896,14690,13589,13444,21639,19999,37300,30491,18845,29934,13403,38726,16109,20373,13293,23020,16069,37503,14798,12955,17166,37711,12515,39926,12485,36751,11243,10373,17779,16428,37098,33688,15970,36399,24300,32671,31452,37965,10965,36322,37145,28342,16346,10536,30244,35751,15633,12008,25872,25201,13094,21556,12405,14736,11069,11920,10163,17071,28711,22907,22873,15532,36526,10646,39418,33151,19043,13117,25954,31716,32356,24568,33244,16766,10193,38824,34000,30163,13299,22939,28337,26660,27540,36657,11177,14722,35610,35490,34838,34941,23082,39348,28479,38578,37050,10356,39422,34034,18782,29420,28733,23660,10630,20273,39058,13738,36599,17247,13834,19405,19337,10526,30257,36908,21298,26618,20604,16373,39839,19648,25708,20680,14755,31446,20510,18687,15262,33224,16151,19388,17910,28972,12365,19453,27058,28348,22619,29730,27707,30731,14371,19164,34020,35663,32525,26836,38604,28037,37012,16550,38446,27682,37589,32693,12712,26526,26785,25795,13234,12708,30460,19117,39692,37429,22974,28906,35178,34391,32286,32291,12486,11387,13520,12112,13884,19634,14076,16807,27945,26070,36955,21220,14401,13466,30254,14125,12241,16950,38650,27793,24017,15497,27442,27430,24361.23828,25292,24066,36603,28888,20187,5777,11472,36929,28287,13153,26950,34394,38030,30548,34696,20640,39773,24742,35476,19187,10085,36150,23184,32090,14495,31207,29536,30832,26998,34204,29310,38321,12447,34846,20299,21330,17755,31495,32239,12563,37631,26917,28653,12762,20913,12327,10571,29504,32498,21265,13766,36488,36061,12383,28843,39340,31865,20575,39715,22359,31444,38754,34235,24967,15986,28108,20356,25162,39581,10567,22402,18473,22802,27928,22598,16487,32844,31518,25018,20107,39121,39963,16935,34807,39834,31109,31517,19763,35910,20285,23624,34170,13874,15547,11885,39786,38037,27598,31205,33236,39417,10638,18607,21218,33890,37372,29756,26655,16284,28334,11348,10269,26918,39853,39648,23490,38843,32109,11393,29908,35361,34327,26807,12503,39922,14411,37640,15568,38102,22263,22958,31126,26585,13675,37627,31411,13958,10003,12245,39669,35466,15807,16200,18200,26479,20207,35682,34898,12425,11191,15712,29344,11411,32641,23393,16731,16456,17703,15216,16851,30890,30172,11205,31258,27497,23675,22734,14147,31521,11582,18341,28941,29399,23341,31102,30911,38115,12291,18882,34566,16960,33817,25627,24940,35374,12292,15963,25797,12164,31248,30901,37094,22009,33404,30417,18554,38957,36969,30505,25771,20545,16020,22096,39354,39205,10323,26116,13838,29219,26891,29126,19676,18653,38143,10273,11606,38508,12971,27722,25435,14783,29057,10940,38303,24680,24979,27542,32642,19901,24557,38023,35665,39179,16073,15689,17005,31993,29202,16461,23310,15110,25882,19589,22978,11362,16924,20560,15669,10642,15654,11644,29323,39929,28004,28938,28449,36366,33329,15578,39784,12416,19010,17914,12871,32132,38876,37396,1531,22804,23733,21306,26015,10311,30913,34936,33907,22724,32537,16815,31386,24493,19095,35531,25092,26147,35725,23852,19641,33462,21677,17161,32745,27626,28483,35101,19663,19923,16477,13962,15294,24476,28511,39227,24418,12684,22887,27659,28977,26582,11605,17832,22929,34801,12250,34541,39811,20355,37481,30667,17738,33858,35995,28939,34099,18202,30155,11964,24942,34909,22810,16973,33323,14871,33066,18730,20040,17116,25420,12662,27306,34791,38021,16986,35043,10610,23741,28065,12937,10381,20698,18898,27580,24747,16814,36194,21171,18235,27930,22212,17390,18796,12872,17206,13391,30766,39382,22601,31800,24046,32882,33213,22503,30634,39819,37511,35859,18491,15769,37223,32726,16282,26903,24833,10790,27886,30292,17412,38764,27960,39276,24036,39576,24105,22876,36376,34387,38979,17002,13775,18058,28696,36674,13361,26225,35790,18680,34985,35017,15802,15587,35868,30063,29983,10680,28020,27153,18956,27258,24201,20820,11327,18742,31167,20144,18234,24007,26900,36170,24912,21128,28376,19610,14479,16219,23777,33015,33954,10517,20309,11281,13040,31541,14273,20076,23027,19767,21581,24239,13922,26465,36788,12105,15327,18559,34028,14624,14299,14886,33099,36826,28259,16038,38210,19939,16648,18386,14823,31533,29412,34313,17920,23700,26792,32817,12074,26276,38190,34218,19359,18496,26440,34661,34018,24019,25125,24461,35216,21930,25068,11834,21894,33525,31901,39847,10791,16923,10137,26819,34003,23215,18442,32289,34579,39719,35919,21147,32990,18542,31347,14636,34195,19711,11798,21418,15647,11161,19288,15979,11724,14656,32571,10192,11960,22585,27686,13343,32003,22332,34446,21401,27709,13975,21297,34984,16813,17567,10064,38693,28735,24120,25587,13247,17287,12351,34092,35827,16097,29981,17752,17266,28353,14555,22081,33076,39376,13386,36019,16641,24960,27890,31045,38899,18298,12919,39530,27470,20951,17384,15244,15655,19991,10609,11918,24123,31034,33332,13002,18909,33018,21126,32431,39579,39944,31632,17007,31747,20960,33079,24751,25103,37070,17899,39954,16888,13409,20535,26796,30176,12991,28774,27061,31948,18984,37491,31369,26294,21380,31549,23473,15932,39799,34510,26290,16856,24766,22219,37525,22533,27482,30843,38080,11551,37823,19808,19968,36869,22994,28948,16566,17509,23516,39315,28404,14049,34996,25346,33938,11404,29594,28177,26335,14434,26118,20523,10732,15253,32955,35651,39827,30395,15168,32816,15994,35363,31196,14849,12994,28717,39522,38868,22675,15471,14355,36652,22560,20348,33307,31910,35432,10841,13363,17718,28375,24083,13616,25617,34179,39629,16150,34530,32829,37870,26802,38994,17453,15103,31336,20529,25135,15429,25464,31377,34607,18086,31195,21469,34167,36920,32157,37687,39416,32432,17819,22916,34624, 33205, 17011, 23594, 16002, 13100, 25067, 20741, 13215, 21853, 20327, 39194, 24065, 25373, 14565, 33254, 18791, 38000, 25650, 35390, 17305, 36329, 16417, 27663, 16285, 18263, 10871, 23878, 14235, 39615, 28272, 15094, 31178]
need_to_fix = [39975,39872,39804,39764,39759,39757,39706,39681,39586,39559,39540,39414,39352,39317,39305,39286,39271,39266,39244,39231,39229,39199,39184,39158,39134,39128,39116,39115,39109,39063,39060,39053,39033,39003,38998,38944,38856,38830,38817,38772,38702,38581,38529,38493,38481,38454,38450,38441,38437,38428,38352,38302,38269,38239,38192,38154,38152,38128,38097,38061,38044,38026,38012,37999,37933,37931,37928,37901,37880,37851,37817,37810,37809,37805,37799,37768,37757,37752,37730,37716,37692,37678,37655,37642,37639,37623,37601,37554,37515,37479,37475,37470,37463,37432,37424,37373,37334,37318,37279,37228,37218,37208,37170,37151,37141,37116,37072,37040,37033,37032,36982,36980,36974,36954,36910,36894,36889,36860,36852,36845,36838,36820,36814,36811,36787,36737,36703,36684,36644,36621,36610,36597,36576,36575,36558,36549,36548,36519,36470,36469,36401,36381,36352,36287,36264,36244,36242,36153,36129,36100,36098,36095,35983,35956,35953,35902,35895,35870,35867,35836,35800,35764,35624,35603,35572,35567,35556,35544,35524,35507,35484,35478,35472,35395,35337,35329,35277,35236,35257,35234,35225,35224,35206,35204,35198,35156,35127,34978,34965,34954,34951,34860,34778,34767,34760,34738,34698,34664,34564,34556,34548,34547,34545,34507,34483,34474,34456,34440,34418,34309,34301,34214,34207,34186,34175,34161,34143,34060,34044.33999,33991,33959,33955,33953,33913,33894,33826,33824,33819,33813,33800,33786,33776,33734,33690,33676,33619,33575,33523,33512,33505,33498,33488,33469,33455,33442,33403,33396,33392,33391,33390,33356,33350,33339,33336,33316,33304,33288,33253,33217,33148,33128,33104,33068,33024,32895,32888,32866,32766,32723,32722,32650,32637,32634,32590,32557,32548,32544,32534,32530,32456,32430,32411,32377,32303,32282,32271,32223,32185,32155,32128,32105,32043,32018,31994,31992,31961,31960,31946,31934,31933,31924,31863,31826,31808,31756,31751,31726,31704,31695,31663,31614,31612,31556,31546,31538,31525,31494,31492,31478,31474,31469,31410,31368,31247,31227,31226,31101,31095,31007,31006,30988,30982,30876,30810,30793,30765,30743,30727,30631,30624,30607,30563,30437,30431,30377,30325,30310,30253,30248,30227,30212,30202,30187,30186,30182,30168,30166,30064,30062,30055,30052,30009,29979,29968,29891,29789,29757,29710,29690,29656,29644,29629,29614,29593,29591,29590,29542,29540,29517,29472,29445,29438,29438,29374,29369,29352,29329,29256,29205,29189,29182,29174,29101,29071,29068,29034,29029,29024,29012,28994,28974,28956,28926,28874,28853,28826,28804,28798,28791,28778,28743,28742,28697,28679,28650,28574,28547,28537,28529,28476,28446,28423,28356,28335,28331,28321,28284,28256,28228,28173,28127,28119,28097,28074,28053,28046,28045,28029,28012,27983,27974,27967,27937,27853,27816,27800,27780,27752,27726,27704,27678,27673,27670,27665,27616,27614,27583,27518,27511,27488,27413,27407,27404,27353,27333,27332,27313,27266,27253,27236,27198,27178,27139,27083,27074,27060,27031,26981,26935,26883,26848,26821,26784,26783,26781,26774,26747,26717,26714,26664,26638,26635,26603,26530,26527,26522,26495,26466,26454,26420,26419,26408,26386,26365,26351,26347,26287,26251,26229,26201,26187,26184,26177,26143,26133,26114,26054,26028,25919,25912,25871,25838,25832,25767,25764,25732,25711,25701,25683,25647,25643,25605,25600,25554,25548,25505,25476,25425,25408,25405,25404,25400,25393,25366,25291,25289,25271,25224,25164,25156,25154,25141,25131,25127,24985,24984,24956,24936,24873,24826,24804,24792,24784,24712,24666,24585,24579,24553,24508,24502,24499,24490,24455,24439,24413,24335,24322,24275,24274,24265,24261,24230,24227,24204,24196,24181,24169,24056,24043,24003,23993,23988,23978,23973,23933,23914,23901,23797,23779,23765,23757,23748,23703,23651,23648,23599,23447,23428,23421,23383,23372,23226,23223,23207,23200,23178,23163,23154,23109,23099,23095,23086,23080,23057,23051,23050,23040,23016,22964,22961,22930,22903,22897,22840,22788,22764,22750,22717,22631,22623,22617,22603,22594,22565,22540,22529,22458,22434,22411,22406,22379,22366,22338,22320,22240,22215,22149,22141,22126,22061,22014,22010,22003,21964,21913,21884,21880,21879,21798,21780,21772,21716,21700,21684,21590,21583,21559,21517,21516,21494,21410,21396,21361,21192,21191,21187,21140,21109,21082,21061,21027,20983,20966,20887,20853,20852,20844,20789,20682,20672,20656,20636,20615,20613,20607,20597,20594,20593,20579,20568,20567,20514,20501,20490,20454,20423,20390,20297,20270,20251,20202,20189,20186,20173,20142,20113,20029,20028,20024,20010,19966,19964,19941,19932,19927,19881,19856,19819,19775,19771,19729,19642,19600,19596,19579,19545,19543,19508,19460,19458,19445,19425,19385,19360,19359,19350,19344,19302,19278,19270,19151,19133,19127,19079,19057,19038,19013,18976,18946,18927,18867,18828,18811,18797,18744,18699,18677,18513,18506,18444,18423,18378,18356,18344,18291,18229,18221,18210,18160,18142,18122,18112,18068,18047,18025,18017,18006,17964,17851,17796,17790,17775,17761,17734,17721,17707,17706,17699,17690,17663,17627,17614,17591,17489,17477,17427,17422,17417,17375,17340,17316,17294,17279,17263,17239,17216,17205,17190,17185,17179,17111,17106,17073,17044,17018,17000,16944,16934,16919,16846,16834,16831,16809,16767,16753,16742,16706,16645,16611,16526,16504,16479,16431,16404,16399,16337,16323,16271,16252,16237,16223,16168,16124,16122,16040,16016,15944,15941,15931,15830,15817,15805,15773,15741,15729,15724,15723,15719,15716,15714,15709,15697,15660,15635,15554,15492,15489,15432,15420,15395,15367,15337,15332,15299,15291,15280,15230,15215,15204,15170,15169,15158,15151,15096,15095,15074,15024,14986,14975,14967,14948,14875,14835,14787,14784,14760,14717,14648,14630,14613,14612,14610,14550,14489,14481,14448,14440,14414,14376,14373,14366,14343,14293,14255,14210,14179,14171,14170,14126,14119,14018,14013,14091,14002,13985,13963,13929,13914,13899,13828,13819,13761,13754,13741,13724,13697,13693,13678,13652,13649,13644,13619,13595,13585,13577,13575,13514,13495,13490,13487,13463,13418,13387,13381,13374,13353,13335,13333,13324,13316,13305,13245,13211,13194,13181,13155,13142,13116,13101,13076,13028,12992,12975,12948,12904,12899,12887,12885,12874,12846,12812,12705,12691,12681,12643,12639,12611,12597,12596,12554,12516,12420,12402,12360,12346,12329,12322,12300,12298,12297,12122,12012,12062,12045,12025,11981,11970,11941,11910,11905,11896,11846,11767,11701,11700,11693,11668,11648,11641,11557,11509,11483,11465,11413,11390,11353,11316,11301,11223,11213,11201,11134,11090,11075,11045,10985,10973,10921,10900,10854,10786,10782,10762,10752,10733,10714,10707,10676,10656,10655,10650,10648,10628,10587,10570,10013,10026,10046,10129,10167,10176,10191,10195,10216,10318,10378,10389,10464,10513,10554,10558]

need_to_fix.sort()
print(need_to_fix)
data_path = 'C:/Users/roei.w/Desktop/machine/Input/validation/'
# old_df=pd.read_csv(data_path + 'train_set.csv')
# image_number = '10118'
img_counter = 2387
skip = True
with open('C:/Users/roei.w/Desktop/machine/Input/validation/train_set.csv', 'r') as csvin, open('C:/Users/roei.w/Desktop/machine/Input/tmp.csv', 'a') as csvout:
    reader = csv.reader(csvin)
    writer = csv.writer(csvout, lineterminator = '\n')
    headers = next(reader)
    print (headers)
    # writer.writerow(headers)
    tagged_cnt = 984
    for row in reader:
        # if(row[0])
        if skip:
            if row[0] == '38093':
                skip = False
                continue
            else:
                continue
        tmp_data = row
        print(tmp_data)
        if int(row[0]) in need_to_fix:
            image_number = row[0]
            image_path = data_path + image_number + '.jpg'
            image=Image.open(image_path)
            plt.imshow(image)
            x=plt.ginput(1)
            # res=(image_number,x)
            # print("x is ",x)
            # result_list.append(res)
            # print(res[0])
            # print(res[1])
            # print(tmp_data[14])
            tmp_data[9]=x[0][0]
            tmp_data[10]=x[0][1]
            print(tmp_data)
            # print(tmp_data)
            tagged_cnt += 1
        print("finished going throgh ", img_counter, " images. last image was: ", row[0]," tagged already: ", tagged_cnt," images")
        print("precent is: ", float(img_counter/2667) , " tagged is: ",float(tagged_cnt/1091), " diffrence: ",float(img_counter/2667)-float(tagged_cnt/1091) )
        writer.writerow(tmp_data)
        # print(tmp_data[13], headers[14])
        img_counter += 1


        # if img_counter == 5:
        #     break
        # new_df=old_df.loc[old_df['name_id'].isin([image_number])]
        # new_df[]
        # print(new_df)