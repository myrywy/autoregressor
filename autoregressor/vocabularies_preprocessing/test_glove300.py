from vocabularies_preprocessing.glove300d import Glove300

import tensorflow as tf
import numpy as np
import pytest
from pytest import approx

# Note: Glove300 is also tested in test_simple_examples TODO: move them, maybe refactor to test one thing at time

def test_word_to_id_and_then_vector():
    tokens_input = tf.constant(["no", "it", "was", "n't", "black", "monday"], dtype=tf.string)
    
    glove = Glove300()
    ids = glove.word_to_id_op()(tokens_input)
    vectors = glove.id_to_vector_op()(ids)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        glove.initialize_embeddings_in_graph(tf.get_default_graph(), sess)
        r_vectors = sess.run(vectors)

    assert r_vectors == approx(np.array(
        [
            [-0.14121, 0.034641, -0.443, -0.093265, -0.010022, -0.069041, 0.16335, -0.12964, 0.0045672, 2.3127, -0.12048, 0.054694, -0.22722, 0.059882, -0.28076, -0.2715, 0.17744, 1.4719, 0.14243, 0.25179, 0.039256, -0.19574, 0.25275, -0.12224, -0.23064, -0.0449, 0.18679, -0.27084, 0.67684, -0.13295, 0.13029, 0.2128, -0.25393, -0.34708, -0.013974, 0.17852, 0.16488, 0.080326, 0.029319, -0.56489, -0.17003, 0.20811, 0.43094, 0.2132, 0.26778, 0.063854, -0.23329, 0.18415, 0.14159, 0.10566, 0.042333, 0.16718, 0.14764, 0.051008, 0.07869, 0.29462, -0.031126, -0.024006, -0.13177, -0.38212, 0.049503, 0.08338, 0.17229, 0.10892, 0.40207, 0.16887, 0.20803, -0.16576, -0.10935, 0.25171, 0.2537, 0.12471, -0.065506, 0.11825, -0.083037, -0.12088, 0.17466, -0.12045, 0.42763, 0.65073, 0.065299, -0.18887, -0.40152, -0.078146, -0.45914, -0.096453, 0.36708, -0.28231, 0.38404, -0.07597, -0.1878, 0.11948, -0.22832, -0.16095, 0.14309, -0.0090158, 0.2809, 0.023625, 0.44597, -0.25256, -0.62236, 0.5481, -0.3839, 0.0094859, 0.2257, -0.99585, -0.28107, 0.067278, -0.10536, -0.049949, -0.025037, 0.070037, -0.14745, -0.053963, 0.37517, -0.31097, 0.10935, -0.12523, -0.031915, -0.43703, -0.12165, 0.09749, 0.073047, 0.049151, -0.21212, -0.15012, -0.022766, -0.30876, 0.028561, -0.10836, 0.069416, -0.10536, -0.16433, -0.32558, 0.50645, 0.13393, -0.21098, -0.029829, 0.093212, -0.45122, -1.6421, 0.078953, 0.35313, -0.28202, 0.26932, -0.0094641, 0.099173, 0.074177, -0.29891, 0.056616, 0.25049, -0.45163, 0.49712, 0.11657, -0.15597, -0.028287, 0.072622, -0.26618, 0.18588, 0.043537, -0.16162, -0.17738, -0.12787, 0.12671, -0.12695, 0.13798, -0.17422, 0.19985, 0.18507, -0.02821, -0.27801, -0.11924, 0.27196, -0.093397, -0.24152, 0.71304, 0.058818, 0.23003, -0.18196, -0.038031, 0.061856, -0.095681, 0.20094, -0.059437, -0.32578, 0.23677, 0.18845, -0.093786, 0.29071, 0.074056, 0.16738, 0.31971, -0.48415, -0.076829, 0.0065072, -0.12015, -0.12628, 0.0627, 0.1041, 0.81267, 0.13162, 0.37337, -0.20733, -0.1292, 0.38116, 0.2539, 0.1688, -0.26463, 0.10273, 0.28119, -0.15295, 0.1548, 0.24093, -0.18426, 0.23231, -0.19153, -0.15871, 0.1979, -0.10288, 0.2818, -0.44362, 0.01553, -0.064855, 0.16203, 0.15307, 0.35869, -0.0072469, -0.056632, -0.18384, -0.020465, -0.059468, -0.1433, 0.3238, -0.16607, -0.014596, 0.36057, 0.54558, -0.34755, -0.22197, 0.034603, 0.089877, 0.66228, -0.094153, 0.17281, 0.043724, -0.23963, 0.0040285, 0.10264, -0.060451, 0.30558, -0.12715, -0.44602, -0.36197, -0.20433, -0.15639, 0.75049, -0.49277, -0.314, 0.23212, 0.14506, -0.10745, 0.26306, -0.13694, 0.49217, -0.3333, 0.13349, 0.33744, -0.04892, 0.7233, -0.035786, 0.36221, 0.28324, -0.18857, -0.0158, 0.19572, 0.14628, -0.0049576, -0.20363, -0.21408, 0.21958, -0.1376, 0.051295, -0.035402, -0.33176, -0.39541, 0.16886, -0.36042, -0.18925, -0.10028, -0.18858, -0.22911, -0.09778, -0.27021, -0.034178, 0.36786, 0.00639, -0.039546, -0.29866, 0.013515, 0.025409],
            [0.0013629, 0.35653, -0.055497, -0.16607, 0.0031402, -0.061926, -0.24759, -0.22897, -0.09105, 2.6751, -0.15062, 0.072403, 0.0061949, -0.0065698, -0.26418, -0.19543, -0.15048, 1.2156, -0.12551, -0.12572, 0.023065, 0.024727, 0.14311, 0.10148, -0.10566, 0.07864, -0.10306, -0.11968, 0.04202, -0.36815, -0.087136, 0.38589, 0.0044597, -0.18259, -0.1226, -0.10454, 0.16039, 0.27415, 0.042427, -0.049497, 0.041286, 0.12223, 0.10821, -0.056199, 0.21754, 0.10983, -0.38878, -0.10935, -0.36647, 0.1342, -0.076634, 0.38148, -0.19979, 0.09391, 0.35189, -0.11133, 0.095313, -0.29593, 0.29022, -0.1966, -0.10331, -0.21995, -0.041991, 0.16631, 0.01523, -0.29185, -0.05472, -0.040665, 0.084861, -0.009206, 0.24625, 0.081873, 0.34256, -0.16768, -0.079394, 0.13206, 0.2156, -0.11199, -0.39589, 0.32299, 0.089602, -0.026041, -0.23981, 0.049861, 0.055241, -0.50554, 0.23002, -0.54613, 0.58194, 0.096957, -0.015559, 0.069833, -0.009668, 0.19936, 0.19006, 0.32913, -0.064844, -0.22404, -0.031196, 0.1818, -0.071896, -0.072126, -0.082155, 0.064145, 0.11215, -1.0712, 0.29581, 0.081019, -0.24954, -0.087734, 0.015893, -0.15779, 0.055281, -0.080948, -0.11288, 0.099631, -0.17395, 0.1223, -0.099904, -0.20707, -0.014483, -0.10597, -0.0031063, 0.04332, -0.20624, 0.25975, -0.12992, -0.32278, 0.17298, -0.15952, -0.19577, -0.22784, 0.022428, 0.19393, -0.059827, -0.011456, 0.17045, -0.041847, -0.1288, -0.067707, -1.9181, 0.29674, 0.27561, 0.29993, 0.17498, -0.25321, -0.2017, 0.13409, -0.049065, 0.13186, -0.10889, 0.23631, 0.17895, 0.024289, 0.0092826, -0.29032, -0.27692, -0.22906, -0.13153, 0.10656, -0.25672, 0.12929, 0.10399, -0.098408, -0.42153, -0.1332, 0.091175, 0.040061, 0.2084, -0.17849, -0.057709, -0.10256, -0.095817, -0.43439, -0.064794, 0.12916, 0.085435, -0.3746, -0.069798, 0.020042, 0.041425, -0.021527, -0.086333, -0.0095633, 0.025466, -0.16101, -0.089574, -0.21178, 0.088594, 0.087381, 0.052047, -0.20386, -0.29424, 0.10097, 0.076137, 0.10431, -0.19752, -0.34268, 0.058982, 0.26035, -0.16364, 0.03294, -0.30368, 0.0734, 0.071074, 0.17129, -0.14442, -0.041817, 0.20912, 0.032747, -0.10649, -0.33475, 0.0088305, -0.32619, 0.21179, 0.29881, -0.013775, -0.090821, -0.33841, -0.1129, 0.12137, 0.059202, -0.12133, -0.093398, 0.15426, -0.032649, -0.11216, 0.28842, -0.036565, -0.041662, -0.22413, 0.060877, 0.21641, 0.30208, 0.16084, -0.027118, 0.26084, -0.090324, -0.1036, -0.01901, 0.34244, 0.025017, 0.025958, 0.21387, 0.43512, -0.67789, -0.0039166, -0.30027, -0.058978, 0.17072, 0.14497, -0.19255, 0.13603, -0.16038, -0.075959, 0.2282, 0.20681, -0.011637, 0.086048, -0.034803, 0.23821, 0.21667, 0.10353, -0.012959, 0.36174, -0.12104, -0.033488, -0.030755, 0.43549, 0.1896, 0.45975, -0.34826, -0.16406, -0.12197, -0.064298, 0.19573, 0.017949, -0.12379, -0.0081198, 0.4002, 0.17065, -0.10712, 0.088398, -0.11473, -0.069708, -0.09321, 0.25621, -0.035815, 0.15968, -0.37266, -0.24035, -0.089325, 0.10603, -0.16025, -0.054419, -0.30824, -0.26249, -0.11237, 0.078259, 0.22398],
            [-0.044058, 0.36611, 0.18032, -0.24942, -0.098095, 0.033261, 0.119, -0.51164, -0.16415, 3.136, 0.20901, 0.29082, 0.25193, -0.020379, -0.24789, -0.47501, -0.038328, 0.56434, -0.038566, -0.11559, 0.024392, -0.45873, -0.10009, 0.21731, 0.16996, -0.12939, 0.0063318, -0.017798, -0.18673, -0.1167, -0.14384, -0.0097187, 0.45289, -0.036453, -0.40523, -0.31816, -0.23389, -0.012272, -0.21479, -0.17841, 0.34474, 0.31133, 0.20543, -0.1896, 0.38995, 0.12103, -0.33685, -0.57051, 0.20732, 0.087872, 0.071458, 0.046355, -0.17425, 0.27856, 0.35989, -0.017122, 0.12197, -0.35806, 0.33181, -0.19827, -0.10386, -0.096699, 0.094231, 0.46722, -0.36612, -0.038628, 0.063485, -0.25765, -0.20415, 0.075931, 0.085753, 0.28176, -0.12443, -0.19756, 0.17218, -0.20121, 0.048154, 0.1301, -0.51096, 0.41643, 0.16487, 0.083688, 0.025331, 0.0014575, 0.26935, -0.46159, 0.18639, -0.6424, -0.2277, 0.032521, 0.050105, 0.1683, -0.27886, -0.037346, 0.50521, -0.39343, 0.25004, -0.091487, 0.044709, 0.15579, -0.19423, 0.29651, -0.27465, -0.33689, -0.11362, -0.43028, 0.016673, -0.015717, 0.15385, -0.30998, -0.17927, -0.002689, -0.029884, -0.18535, -0.079747, -0.31545, 0.0024644, 0.19685, -0.061948, -0.2724, 0.0372, 0.24951, 0.15755, -0.084023, -0.24132, 0.35744, -0.16309, -0.67866, -0.27942, -0.016828, 0.017248, -0.060522, -0.26155, 0.16951, 0.50993, -0.46213, -0.019627, 0.3955, 0.0053794, -0.13616, -1.3947, 0.24283, 0.33351, 0.18875, 0.33386, -0.1979, -0.45546, -0.14531, 0.32496, -0.24984, -0.38316, -0.047484, 0.3163, -0.27841, -0.31328, -0.13258, -0.15671, 0.050417, 0.2073, -0.13118, -0.40559, -0.34316, 0.14348, -0.45976, -0.48611, -0.32394, -0.19056, 0.16412, 0.22827, -0.054174, 0.039441, 0.079182, -0.034827, -0.043719, -0.56115, -0.18462, 0.012758, -0.058201, -0.4096, -0.28184, -0.035173, -0.27668, -0.44195, -0.094452, -0.36051, -0.23688, -0.22469, 0.22704, 0.070153, 0.079784, -0.050581, 0.19954, -0.53252, 0.38514, 0.20942, 0.3133, 0.37957, -0.31456, -0.22611, -0.14732, 0.12792, 0.026238, -0.19538, 0.2053, 0.18387, 0.070116, 0.20402, -0.057152, 0.16134, 0.023932, 0.04476, -0.0031943, 0.0076469, -0.032653, 0.39232, 0.11799, -0.18832, -0.21732, -0.038809, -0.19023, -0.067095, 0.021589, 0.03139, 0.27935, -0.25991, -0.010694, 0.071357, 0.20587, 0.030717, 0.14273, -0.012696, 0.30787, 0.1761, -0.23735, 0.10864, -0.34518, 0.051447, 0.060717, -0.050337, -0.018071, -0.39068, -0.0020948, 0.21507, 0.30334, 0.079873, -0.135, -0.0033115, -0.43378, 0.14857, -0.028767, -0.091394, -0.11293, 0.14341, -0.02577, 0.3054, -0.56747, 0.30705, -0.085973, -0.021836, 0.14566, 0.57363, 0.27721, -0.25141, 0.12354, 0.0045573, 0.10348, 0.14283, 0.086515, -0.11795, 0.070627, 0.455, 0.14827, -0.33691, -0.26387, -0.40101, -0.034913, 0.032671, -0.42077, 0.058225, 0.38307, 0.59657, 0.33333, 0.025108, -0.10701, 0.030241, -0.079168, -0.02454, 0.24922, 0.061272, 0.012772, -0.019862, 0.082316, 0.49588, 0.09668, 0.43798, 0.062743, -0.053951, 0.18625, -0.097817, -6.7104e-05],
            [-0.13019, 0.27764, -0.24159, -0.1229, -0.099673, 0.1548, 0.045905, -0.048852, -0.070227, 2.6683, 0.018325, -0.14477, 0.54896, 0.13912, -0.5663, -0.076259, 0.020997, 0.53276, -0.31074, 0.12863, 0.21444, 0.13455, 0.2385, 0.048554, -0.3388, 0.021484, -0.3094, -0.298, 0.54086, -0.53253, -0.26099, 0.26757, -0.056835, -0.15544, 0.27707, 0.085732, 0.32584, 0.088359, -0.1003, -0.13693, -0.0411, 0.23295, -0.091702, 0.081552, 0.20786, -0.031941, -0.21681, 0.042795, -0.11452, 0.059274, -0.22558, -0.11495, 0.1403, -0.10689, 0.41254, -0.10124, 0.02884, -0.41875, 0.10929, -0.15641, 0.039418, -0.28243, -0.1422, 0.47574, 0.26084, -0.12307, -0.1724, -0.10748, 0.31673, 0.25172, 0.36466, 0.38725, 0.28571, -0.093661, 0.055286, 0.2427, 0.17109, -0.15716, 0.14724, 0.36599, -0.065697, -0.23486, -0.19004, -0.042001, -0.2401, -0.17693, 0.14648, -0.5474, 0.7324, -0.15878, -0.34187, -0.22201, -0.15924, 0.31443, 0.23653, 0.055091, 0.28245, 0.019587, 0.2493, 0.044079, -0.22886, 0.36507, -0.076332, -0.073083, 0.23082, -0.47279, 0.020098, -0.18732, -0.15627, 0.29637, 0.081997, -0.019479, 0.31497, -0.26458, 0.17927, -0.11421, 0.1828, -0.19653, -0.15288, -0.0028553, -0.020426, -0.22377, -0.070952, -0.36273, 0.25286, -0.017657, 0.070154, -0.52035, -0.06306, -0.063401, 0.031118, 0.17004, -0.12248, 0.344, 0.16571, -0.15052, 0.052269, -0.10966, -0.1355, -0.23961, -2.2062, 0.14258, 0.16897, 0.035696, 0.1303, -0.10539, -0.2026, 0.12489, -0.01408, -0.22252, 0.40705, 0.1008, 0.032245, 0.39523, -0.077507, -0.21702, -0.09814, -0.20212, -0.019024, 0.012406, -0.33449, 0.24188, -0.2918, 0.047731, -0.2364, -0.041121, 0.16712, -0.25078, 0.30386, -0.15052, 0.025404, -0.19863, 0.11755, -0.35061, 0.26133, 0.37007, 0.13698, 0.11351, 0.15074, -0.1246, -0.067939, 0.01238, -0.42154, -0.0091529, -0.10902, -0.086627, -0.059834, 0.10847, 0.013728, 0.043898, 0.16817, 0.043819, -0.29907, -0.17329, 0.21429, 0.13766, -0.088914, -0.22155, 0.34939, 0.48677, -0.13624, -0.086712, 0.027144, -0.24487, 0.043639, 0.13863, 0.012576, -0.091621, 0.35802, 0.084971, 0.019817, -0.20481, -0.15568, -0.52826, 0.064368, 0.060007, -0.24596, 0.15001, -0.60509, -0.15379, 0.016719, 0.15949, -0.097624, -0.07951, 0.19468, 0.098393, 0.15924, 0.15078, -0.091977, -0.0268, -0.21947, -0.19844, 0.58255, 0.074457, -0.18159, 0.10575, 0.39545, -0.62542, -0.070905, 0.29926, 0.23098, -0.058532, 0.079409, 0.21516, -0.0001531, -0.35988, -0.2874, -0.024583, -0.30688, 0.61378, -0.2017, -0.18743, -0.22567, -0.11391, -0.13839, 0.065526, -0.38524, -0.26101, 0.15433, 0.19295, 0.016614, 0.27393, 0.015089, 0.17114, 0.36981, 0.44011, -0.0013757, 0.15895, 0.52241, 0.21377, 0.099801, -0.07774, -0.02571, -0.45929, 0.009682, -0.015693, -0.012355, -0.15352, 0.040929, 0.24291, -0.099217, -0.12023, 0.048583, -0.22753, -0.40505, -0.23716, -0.011524, -0.15346, 0.068119, -0.035336, -0.34307, 0.065718, -0.026112, 0.083108, 0.27713, 0.020035, -0.20193, -0.17143, 0.55838, 0.19698],
            [-0.29365, -0.049916, 0.096439, -0.089388, 0.27109, 0.057496, -0.50298, 0.11331, -0.19913, 1.0869, -0.36474, 0.18028, -0.2439, -0.84879, -0.09803, 0.22358, 0.16649, 1.8263, -0.30784, -0.45779, -0.13423, -0.7684, 0.061036, 0.13364, -0.07578, -0.36814, -0.56498, 0.11553, 0.18909, 0.069852, 0.10334, 0.54858, -0.017279, -0.42885, 0.17587, -0.48115, -0.21931, -0.39983, -0.05173, -0.46209, 0.46579, 0.21905, -0.14852, 0.11248, 0.21266, -0.13285, -0.1344, 0.22768, 0.38002, -0.31141, -0.75913, 0.34262, -0.57856, -0.44662, 0.17095, 0.13949, -0.28634, 0.066538, -0.21849, -0.48396, -0.73416, -0.45858, 0.20657, 0.0091145, -0.0039049, 0.01489, -0.25298, -0.022714, -0.027294, 0.41785, 0.11382, -0.33901, -0.032653, 0.042876, -0.1628, -0.083524, -0.36741, -0.26457, 0.053942, -0.01116, -0.50069, -0.16943, 0.10525, -0.030164, 0.4385, -0.13928, 1.141, 0.76126, 0.074075, -0.028966, 0.066959, 0.20611, 0.27884, -0.17062, 0.0044823, -0.46235, -0.052986, 0.50416, -0.018854, -0.38912, 0.57516, 0.61789, 0.45961, -0.1963, -0.51927, -0.51316, -0.8881, 0.28339, 0.032175, 0.26376, -0.47802, -0.35921, -0.50878, -0.1828, 0.26999, 0.24097, 0.099165, -0.031377, 0.089655, 0.32511, -0.42431, 0.01075, -0.32665, 0.15986, 0.16415, 0.38453, 0.24862, -0.31164, 0.16802, -0.38192, 0.092993, -0.033324, -0.13209, 0.038213, -0.0029631, 0.06452, 0.0079986, -0.50266, -0.018759, 0.05632, -3.0279, -0.079183, 0.70083, 0.2262, 0.36396, -0.096987, 0.19656, 0.012033, 0.23194, -0.030562, -0.28404, -0.37286, -0.005297, -0.33137, -0.44292, 0.28554, -0.71202, -0.0015515, 0.0093941, 0.31106, -0.20186, -0.10606, -0.0098406, 0.083881, 0.0014653, -0.43426, -0.13004, -0.14525, 0.24627, -0.038385, -0.33198, 0.4009, -0.053365, 0.47144, -0.18795, 0.25009, -0.22505, 0.10527, 0.4418, 0.18197, -0.4826, 0.51301, -0.21059, -0.51911, -0.18121, 0.69244, -0.36925, 0.13242, 0.17995, 0.024023, -0.092837, -0.16256, -0.25677, 0.058971, 0.4761, -0.12983, 0.019869, 0.22802, -0.36084, -0.091776, 0.45292, -0.027555, -0.15405, -0.30351, 0.16619, -0.074507, 0.12211, -0.14763, -0.1045, 0.39327, -0.058905, 0.6207, -0.49493, 0.023326, 0.37233, 0.032352, -0.65445, -0.32216, 0.39367, -0.12799, -0.78568, -0.13649, -0.59398, -0.039309, -0.16203, -0.088509, 0.14446, -0.14543, 0.17516, 0.67057, -0.31062, -0.31735, 0.48737, 0.51206, 0.12244, 0.58553, -0.3483, -0.070485, 0.65111, 0.49588, -0.042622, 0.085238, -0.24129, -0.61676, 0.065639, 0.21727, -0.31657, -0.20381, -0.18905, -0.0026379, -0.16428, 0.29292, -0.043597, -0.10713, 0.015803, 0.10977, 0.099193, 0.058263, -0.22138, 0.53114, 0.2194, 0.46687, -0.22339, 0.45082, -0.34546, -0.10945, 0.013951, -0.22981, -0.61019, 0.53618, -0.38039, -0.3018, 0.044355, -0.47215, 0.094294, -0.30885, -0.16255, 0.35686, -0.0010873, -0.13689, -0.24389, 0.64798, 0.19567, -0.17806, -0.46973, -0.026857, 0.25365, 0.099388, 0.057244, -0.32616, -0.59946, -0.070698, 0.044969, -0.83205, -0.37187, 0.28149, 0.1978, 0.047221, -0.22288, 0.017735],
            [0.031091, 0.56825, -0.03107, 0.004301, -0.03025, -0.2201, 0.016359, -0.27483, 0.54576, 0.69811, -0.92913, -0.32617, 0.34225, -0.36393, 0.17427, 0.10333, -0.22877, 0.62709, 0.40462, -0.27718, 0.051787, 0.26553, 0.0028972, -0.37731, -0.092281, 0.26781, -0.51189, -0.34465, -0.089694, -0.13627, -0.073969, -0.23845, 0.4223, -0.2777, -0.68097, 0.63363, 0.084718, 0.27264, -0.24687, -0.0064634, -0.19284, 0.022436, -0.15938, 0.50185, -0.5157, 0.43452, -0.07777, -0.14402, 0.21616, -0.11667, 0.013378, 0.12769, 0.31937, 0.052952, 0.20743, 0.25537, -0.42906, 0.058272, 0.24935, -0.38469, -0.70756, -0.28694, 0.21439, -0.49421, 0.071887, 0.028562, -0.399, 0.43777, -0.15642, 0.39649, -0.17438, -0.23927, 0.038615, 0.2249, 0.72184, 0.36085, 0.0488, 0.73647, -0.155, 0.57072, -0.040056, 0.27431, -0.12806, -0.1526, 0.14863, -0.0065193, 1.0662, -0.36083, 0.70706, 0.043639, 0.47063, 0.027411, -0.23215, -0.46128, -0.049949, 0.020991, -0.041238, 0.43573, 0.30043, 0.35162, -0.29331, -0.68115, 0.25255, 0.23526, -0.13769, -0.62365, 0.5771, -0.13667, -0.025982, -0.0593, -0.016144, -0.67935, 0.18109, 0.055385, -0.16223, 0.43419, 0.23547, 0.1063, 0.14108, 0.13307, 0.58862, 0.091575, -0.18413, 0.19386, 0.1448, -0.095191, 0.19922, 0.075761, -0.019164, -0.077082, 0.23526, -0.80003, -0.27262, 0.23965, -0.026556, -0.089179, -0.0030789, -0.20901, -0.13612, -0.22583, -2.2758, -0.18385, -0.16255, -0.42371, 0.15869, 0.35939, 0.074838, 0.24797, -0.05182, -0.26273, -0.11361, -0.42738, 0.083827, -0.20052, 0.13626, -0.28534, 0.22639, -0.32933, -0.11853, 0.085149, -0.003949, -0.35466, -0.23401, -0.13937, 0.0068301, -0.12079, -0.1791, 0.73794, -0.24173, 0.18064, -0.044553, 0.094685, -0.3187, 0.0014406, -0.42592, 0.29623, -0.17538, -0.51084, -0.041933, 0.18008, -0.16072, -0.062, -0.2527, 0.35847, 0.18067, -0.11788, -0.038845, 0.24696, 0.16047, -0.0051667, 0.24596, -0.24756, -0.15568, -0.37533, 0.098757, 0.64188, -0.0036217, -0.050042, 0.2193, -0.29602, -0.20227, 0.21918, 0.36836, 0.29122, 0.16094, -0.70123, -0.49206, 0.32254, 0.26288, 0.012186, 0.32156, 0.49782, 0.0039978, 0.36138, -0.27197, -0.30703, 0.093815, -0.76536, -0.3478, 0.48628, 0.44034, -0.62981, -0.49056, 0.12199, -0.15922, -0.081561, -0.14558, -0.16878, -0.37092, -0.33377, -0.29117, 0.591, 0.052894, -0.028036, -0.10446, -0.37111, -0.38053, -0.22004, -0.22515, 0.25515, -0.11202, -0.5538, 0.086523, 0.48785, 0.072203, 0.29461, 0.23643, -0.11222, -0.092494, -0.28043, -0.13144, 0.234, 0.11143, -0.67456, 0.43617, 0.023155, -0.57365, -0.39816, -0.73945, -0.44138, 0.21267, -0.018604, -0.25674, -0.025934, -0.23015, -0.25172, -0.40583, -0.083189, -0.054541, 0.15206, -0.31548, -0.14732, -0.23183, 0.75427, 0.018009, 0.30702, -0.22941, -0.013627, 0.38182, 0.26575, 0.63149, -0.88812, -0.6673, 0.080639, 0.23583, 0.49035, 0.10201, -0.16356, -0.11952, 0.60617, 0.38027, -0.0078457, 0.039968, -0.007106, 0.32, -0.26781, 0.41864, 0.12264, -0.43825, 0.090428],
        ]
    ))