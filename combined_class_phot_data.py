import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import re
import matplotlib as mpl
import time
import json
import pandas
import math
#from get_class_data import get_totals, 


class FullSubject:
    def __init__(self, zooniverse_id, wise_id, class_data_vec, phot_data_vec):
        self.zooniverse_id = zooniverse_id
        self.wise_id = wise_id
        self.num_classifiers = float(class_data_vec[0])
        self.classification_dict = {}
        self.classification_dict[u'good'] = float(class_data_vec[1])
        self.classification_dict[u'multi'] = float(class_data_vec[2])
        self.classification_dict[u'oval'] = float(class_data_vec[3])
        self.classification_dict[u'empty'] = float(class_data_vec[4])
        self.classification_dict[u'extended'] = float(class_data_vec[5])
        self.classification_dict[u'shift'] = float(class_data_vec[6])
        self.state = class_data_vec[7]
        self.jmag = phot_data_vec[4]
        self.jmagerr = phot_data_vec[5]
        self.hmag = phot_data_vec[6]
        self.hmagerr = phot_data_vec[7]
        self.kmag = phot_data_vec[8]
        self.kmagerr = phot_data_vec[9]
        self.w1mag = phot_data_vec[10]
        self.w1magerr = phot_data_vec[11]
        self.w2mag = phot_data_vec[12]
        self.w2magerr = phot_data_vec[13]
        self.w3mag = phot_data_vec[14]
        self.w3magerr = phot_data_vec[15]
        self.w4mag = phot_data_vec[16]
        self.w4magerr = phot_data_vec[17]
        self.w4xs = self.w1mag - self.w4mag
        self.w4xserr = np.sqrt((self.w1magerr**2) + (self.w4magerr**2))
        self.w3xs = self.w1mag - self.w3mag
        self.w3xserr = np.sqrt((self.w1magerr**2) + (self.w3magerr**2))

        self.JminH = self.jmag - self.hmag
        self.HminK = self.hmag - self.kmag
       

        self.eq_coords = [phot_data_vec[0], phot_data_vec[1]]
        self.gal_coords = [phot_data_vec[2], phot_data_vec[3]]

        #self.wise_id = subject_dict[u'metadata'][u'wise_id']
        #if len(subject_dict[u'coords']) > 0:
        #    self.eq_coords = subject_dict[u'coords']
        #else:
        #    get_coords_string = subject_dict[u'metadata'][u'wise_id']
        #    self.eq_coords = get_eq_coords(get_coords_string)
        #self.eq_coords = subject_dict[u'coords']
        #self.gal_coords = get_gal_coords(self.eq_coords)
        #self.jmag = jmag
        #self.jmagerr = jmagerr

        #self.eq_coords = None
        #self.gal_coords = None
        self.majmult = False
        self.majgood = False
        self.other = False
        if self.num_classifiers > 0:
            if ((self.classification_dict[u'good']/self.num_classifiers) > 0.5):
                self.majgood = True
            elif ((self.classification_dict[u'multi']/self.num_classifiers) > 0.5):
                self.majmult = True
            else:
                self.other = True

    def __str__(self):
        printlist = []
        printlist.append(str(self.zooniverse_id))
	printlist.append(str(self.wise_id))
        printlist.append('['+str(self.eq_coords[0])+','+str(self.eq_coords[1])+']')
        printlist.append('['+str(self.gal_coords[0])+','+str(self.gal_coords[1])+']') 
        printlist.append(str(self.num_classifiers))
        printlist.append(str(self.classification_dict[u'good']))
        printlist.append(str(self.classification_dict[u'multi']))
        printlist.append(str(self.classification_dict[u'oval']))
        printlist.append(str(self.classification_dict[u'empty']))
        printlist.append(str(self.classification_dict[u'extended']))
        printlist.append(str(self.classification_dict[u'shift']))
        printlist.append(str(self.state))
        printlist.append(str(self.jmag))
        printlist.append(str(self.jmagerr))
        printlist.append(str(self.hmag))
        printlist.append(str(self.hmagerr))
        printlist.append(str(self.kmag))
        printlist.append(str(self.kmagerr))
        printlist.append(str(self.w1mag))
        printlist.append(str(self.w1magerr))
        printlist.append(str(self.w2mag))
        printlist.append(str(self.w2magerr))
        printlist.append(str(self.w3mag))
        printlist.append(str(self.w3magerr))
        printlist.append(str(self.w4mag))
        printlist.append(str(self.w4magerr))
   
        #printlist.append("jmag : " + str(self.jmag) + " +/m " + str(self.jmagerr))
        #if self.majmult is True:
        #    printlist.append("Maj Mult")
        #elif self.majgood is True:
        #    printlist.append("Maj Good")
        #else:
        #    printlist.append("Other")

        s = ""
        for entry in printlist:
            s = s + entry + ","

        s=s[:-1]

        return s


def photdriver(classification_output_filename, combined_photometry_filename, directory):
    start_time = time.time()    
    print start_time

    class_dict = {}
    #phot_list = []
    phot_dict = {}
    zoo_wise_dict = {}
    object_list = []
    wise_zoo_dict = {}

    #with open(classification_output_filename) as f:
    #    for line in f:
    #        class_data_vec = line.split(",")
    #        zoo_id = class_data_vec[0]
    #        wise_id = class_data_vec[1]
    #        class_dict[zoo_id] = class_data_vec[2:]
    #        zoo_wise_dict[zoo_id] = wise_id
    #        if wise_id not in wise_zoo_dict.keys():
    #            wise_zoo_dict[wise_id] = [zoo_id]
    #        else:
    #            wise_zoo_dict[wise_id].append(zoo_id)
    #        if (len(class_dict.keys())%100) < 1:
    #            print len(class_dict.keys())

    df = pandas.read_csv(classification_output_filename)
    data = df.values

    zoo_id_vec = data[:,0].tolist()
    #wise_id_vec = data[:,1].tolist()
    class_dict = dict(zip(zoo_id_vec, data[:,2:]))

    wise_ids = set(data[:,1])
    wise_zoo_dict = {k: [] for k in wise_ids}

    loop_start_time = time.time()
    iter_count = 0
    for line_vec in data:
        #zoo_id = line_vec[0]
        #wise_id = line_vec[1]i
        #line = line_vec.tolist()
        #class_dict[line_vec[0]] = line_vec[2:]
        #if line_vec[1] not in wise_zoo_dict.keys():
        #    wise_zoo_dict[line_vec[1]] = [line_vec[0]]
        wise_zoo_dict[line_vec[1]].append(line_vec[0])
        iter_count += 1
        if (iter_count%1000) < 1:
            print iter_count
        if (iter_count%10000) < 1:
            print 'average loop time', (time.time() - loop_start_time)/len(wise_zoo_dict.keys())

    first_read_time = time.time() - start_time
    print first_read_time

    #with open(combined_photometry_filename) as f:
    #    get_num_entries_vec = f.readlines()

    #num_entries = len(get_num_entries_vec)
    #print num_entries

    #phot_list = [None] * num_entries
    #print len(phot_list)
    #iter_count_build = 0
    #iter_count_objects_read = 0

    #wise_zoo_dict = {k: [] for k in wise_ids}
    wise_phot_dict = {k: [] for k in wise_ids}

    #with open(combined_photometry_filename) as f:
    #    build_start_time = time.time()
    #    for line in f:
    #        load = json.loads(line)
    #        #phot_list[i] = (load)
    #        wise_id = load['wiseid']

    #        if wise_id in wise_phot_dict.keys():
    #            #zoo_ids = wise_zoo_dict[wise_id]
    #            #for zoo_id in zoo_ids:
    #            #    object_list.append(FullSubject(zoo_id, wise_id, class_dict[zoo_id], load))
    #            wise_phot_dict[wise_id] = load
    #            iter_count_objects_read += 1
    #            if (iter_count_objects_read % 100) < 1:
    #                print 'Objects read', iter_count_objects_read
    #            #if (iter_count_objects_read % 20000) < 1 and iter_count_objects_read > 0:
    #                print 'average read time', (time.time() - build_start_time)/iter_count_objects_read

    #        iter_count_build += 1
    #        if (iter_count_build % 100) < 1:
    #            print iter_count_build
    #        if (iter_count_build % 20000) < 1:
    #            print 'Average iter time', (time.time() - build_start_time)/iter_count_build
    #        #if len(object_list) > 0:
    #        #    if len(object_list) < 2:
    #        #        print str(object_list[0])
    #        #    if (len(object_list)%100) < 1:
    #        #        print len(object_list)
    #        #        print 'average build time', (time.time() - build_start_time)/len(object_list)
    #        #phot_dict[wise_id] = i
    #        #if (len(phot_dict.keys()) % 100) < 1:
    #        #    print len(phot_dict.keys())
    #        #i += 1
    #        #iter_count_build += 1
    #        #if (iter_count_build % 100) < 1:
    #        #    print 'iters', iter_count_build
    #        #if (iter_count_build % 20000) < 1:
    #        #    print 'average iter time', (time.time() - build_start_time)/iter_count_build

    df1 = pandas.read_csv(combined_photometry_filename)
    data1 = df1.values

    phot_wise_id_vec = data1[:,0].tolist()
    #wise_id_vec = data[:,1].tolist()
    phot_dict = dict(zip(phot_wise_id_vec, data1[:,1:]))

    print 'Photometry read', time.time()-start_time
  
    print phot_wise_id_vec[0], phot_dict[phot_wise_id_vec[0]]    

    #wise_ids = set(data[:,1])
    #wise_zoo_dict = {k: [] for k in wise_ids}

    #loop_start_time = time.time()
    #iter_count = 0
    #for line_vec in data:
        #zoo_id = line_vec[0]
        #wise_id = line_vec[1]i
        #line = line_vec.tolist()
        #class_dict[line_vec[0]] = line_vec[2:]
        #if line_vec[1] not in wise_zoo_dict.keys():
        #    wise_zoo_dict[line_vec[1]] = [line_vec[0]]
    #    wise_zoo_dict[line_vec[1]].append(line_vec[0])
    #    iter_count += 1
    #    if (iter_count%100) < 1:
    #        print iter_count
    #    if (iter_count%20000) < 1:
    #        print 'average loop time', (time.time() - loop_start_time)/len(wise_zoo_dict.keys())

    #first_read_time = time.time() - start_time
    #print first_read_time
    
    get_object_count_start_time = time.time()

    print 'Getting Number Objects', time.time() - start_time

    iter_count = 0
    printed = False
    num_objects = 0

    #for wiseid in wise_ids:
    #    if wiseid in phot_wise_id_vec:
    #        zoo_ids = wise_zoo_dict[wiseid]
    #        for zoo_id in zoo_ids:
    #            num_objects += 1
    #            if (num_objects % 20000) <1:
    #                print (time.time() - get_object_count_start_time)/num_objects

    wiseids_in_photometry = list(set(wise_ids).intersection(phot_wise_id_vec))
    has_paper_two = False
    if 'J080822.18-644357.3' in wiseids_in_photometry:
        print "AWI0005x3s is in"
        has_paper_two = True
    else:
        print "AWI0005x3s is not in"

    num_objects = sum(len(wise_zoo_dict[key]) for key in wiseids_in_photometry)

    object_list = [None for var in range(num_objects)]
    list_index = 0

    print 'Object List Built', num_objects, time.time()-get_object_count_start_time

    new_build_start_time = time.time()

    object_map = {k: [] for k in wiseids_in_photometry}

    for wiseid in wiseids_in_photometry:
        zoo_ids = wise_zoo_dict[wiseid]
        for zoo_id in zoo_ids:
            object_list[list_index] = FullSubject(zoo_id, wiseid, class_dict[zoo_id], phot_dict[wiseid])
            #object_list.append(FullSubject(zoo_id, wiseid, class_dict[zoo_id], phot_dict[wiseid]))
            #if (len(object_list) % 100) < 1:
            #    print len(object_list)
            if object_list[list_index].num_classifiers > 0. and not printed:
                print str(object_list[list_index])
                print object_list[list_index].majmult, object_list[list_index].majgood, object_list[list_index].other
                printed = True
            object_map[wiseid].append(list_index)
            list_index += 1
            if (list_index % 20000) < 1:
                print (time.time() - new_build_start_time)/list_index
        #if wiseid == 'J080822.18-644357.3':
        #    print str(object_list[list_index])
        iter_count += 1
        if (iter_count% 1000) < 1:
            print 'iter_count', iter_count
        if (iter_count % 10000) < 1:
            print (time.time() - new_build_start_time) / list_index
            print 'objects:', list_index


    cur_time = time.time()
    read_time = cur_time - start_time
    print "read and build time", read_time

    if has_paper_two:
        print str(object_list[object_map['J080822.18-644357.3'][0]])

    #for zooniverse_id in zoo_wise_dict.keys():
    #    wise_id = zoo_wise_dict[zooniverse_id]
    #    object_list.append(FullSubject(zooniverse_id, wise_id, class_dict[zooniverse_id], phot_list[phot_dict[wise_id]]))
    #    if (len(object_list)%100) < 1:
    #        print len(object_list)

    brightness_limited_list = brightness_limit(object_list, 14.5)
    print len(brightness_limited_list)

    brightness_limited_completed_list, brightness_limited_remaining_list = get_complete(brightness_limited_list)
    print len(brightness_limited_completed_list)

    blc_goodlist, blc_multilist, blc_otherlist = get_totals(brightness_limited_completed_list)

    print len(blc_goodlist), len(blc_multilist), len(blc_otherlist)

    blc_good_rates, blc_good_test_rate_errs, blc_multi_rates, blc_multi_test_rate_errs = coord_number_histogram(blc_goodlist, blc_multilist, blc_otherlist, directory, 'full_data_set')

    #density_histogram_status = coord_density_histogram(rates, rate_errs, directory, False, 'full_data_set')
    good_density_histogram_status_test = coord_density_histogram(blc_good_rates, blc_good_test_rate_errs, directory, True, 'full_data_set', 'Good Fraction')
    multi_density_histogram_status_test = coord_density_histogram(blc_multi_rates, blc_multi_test_rate_errs, directory, True, 'full_data_set', 'Multiple')

    #both_density_histogram_status_test = coord_density_histogram_mult(blc_good_rates, blc_good_test_rate_errs, blc_multi_rates, blc_multi_test_rate_errs, directory, True, 'full_data_set')

    lat_long_heatmap_status = lat_long_heatmap(blc_goodlist, blc_multilist, blc_otherlist, directory,'full_data_set')

    lat_long_remaining_status = lat_long_remaining(brightness_limited_remaining_list,directory)

    minJmags = min([o.jmag for o in brightness_limited_completed_list])
    Jmagfloor = float(math.floor(minJmags*2.))

    bright_rates, bright_rate_errs, test_bright_rate_errs = bright_number_histogram(blc_goodlist, blc_multilist, blc_otherlist, directory, Jmagfloor)

    bright_density_histogram_status = bright_density_histogram(bright_rates, bright_rate_errs, directory, False, Jmagfloor)
    bright_density_histogram_status_test = bright_density_histogram(bright_rates, test_bright_rate_errs, directory, True, Jmagfloor)

    bright_density_latlong_status = bright_density_latlong(blc_goodlist, blc_multilist, blc_otherlist, directory, Jmagfloor)

    bright_density_latbright_status = bright_density_latbright(blc_goodlist, blc_multilist, blc_otherlist, directory, Jmagfloor)

    #Mstar_color_color_status = Mstar_color_color(blc_goodlist, directory, Jmagfloor)

    num_good_after_background = get_background_galaxies(blc_goodlist)
    print 'Good after background galaxies', num_good_after_background
    print 'Final good fraction', num_good_after_background/len(brightness_limited_completed_list)

    end_time = time.time()
    run_time = end_time - start_time

    print "Done"
    print end_time
    print run_time


#def Mstar_color_color(blc_goodlist, directory, Jmagfloor):


def brightness_limit(subjects, limit):
    return_list = []

    for subject in subjects:
        if (subject.jmag <= limit):
            return_list.append(subject)

    return return_list


def get_complete(subjects):
    return_list_complete = []
    return_list_remaining = []

    for subject in subjects:
        if subject.state == "complete" and subject.num_classifiers > 0:
            return_list_complete.append(subject)
        else:
            return_list_remaining.append(subject)

    return return_list_complete, return_list_remaining

def get_totals(subject_vec):
    goodlist = []
    multilist = []
    otherlist = []

    #tester_limit = 0

    for entry in subject_vec:
        multifrac = float(entry.classification_dict[u'multi']) / float(entry.num_classifiers)
        goodfrac = float(entry.classification_dict[u'good']) / float(entry.num_classifiers)
        if goodfrac > 0.5:
            entry.majgood = True
	    goodlist.append(entry)
        elif multifrac > 0.5:
            entry.majmult = True
            multilist.append(entry)
        else:
            entry.other = True
            otherlist.append(entry)
        #print entry.zooniverse_id, entry.num_classifiers, entry.classification_dict, multifrac, goodfrac, entry.majmult, entry.majgood, entry.other
        #tester_limit += 1
        #if tester_limit > 5000:
        #    break

    return goodlist, multilist, otherlist


def coord_number_histogram(goodlist, multilist, otherlist, directory, strkey):
    #build lists
    goodlat = []
    goodlong = []
    for item in goodlist:
        goodlong.append(item.gal_coords[0])
        goodlat.append(item.gal_coords[1])

    multilat = []
    multilong = []
    for item in multilist:
        multilong.append(item.gal_coords[0])
        multilat.append(item.gal_coords[1])
    
    otherlat = []
    otherlong = []
    for item in otherlist:
        otherlong.append(item.gal_coords[0])
        otherlat.append(item.gal_coords[1])
    
    print len(goodlat), len(multilat), len(otherlat)

    binvec_lat = np.arange(-90., 91., 5.)

    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}

    plt.figure()
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams.update(params)

    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(right=0.95)
    plt.gcf().subplots_adjust(top=0.95)
    plt.gcf().subplots_adjust(bottom=0.12)

    print binvec_lat

    plt.hist([goodlat, multilat, otherlat], bins=binvec_lat, stacked=True, label=('Good', 'Multi', 'Other'))
    #plt.yscale('log', nonposy='clip')
    counts_good,edges_good = np.histogram(goodlat, bins=binvec_lat)
    counts_multi,edges_multi = np.histogram(multilat, bins=binvec_lat)
    counts_other,edges_other = np.histogram(otherlat, bins=binvec_lat)

    f1 = open(directory+'/raw_numbers_binned_'+strkey+'.dat','w')
    for i in range(0, len(binvec_lat)-1, 1):
        f1.write(str(counts_good[i])+','+str(counts_multi[i])+','+str(counts_other[i])+'\n')
    f1.close()

    totals = np.zeros(len(binvec_lat) - 1)
    good_rates = np.zeros(len(binvec_lat) - 1)
    #good_rate_errs = np.zeros(len(binvec_lat) - 1)
    good_test_rate_errs = np.zeros(len(binvec_lat) - 1)
    multi_rates = np.zeros(len(binvec_lat) - 1)
    #multi_rate_errs = np.zeros(len(binvec_lat) - 1)
    multi_test_rate_errs = np.zeros(len(binvec_lat) - 1)


    for i in range(0,len(binvec_lat)-1,1):
        #multi_errs = np.sqrt(float(counts_multi[i]))
        totals[i] += float(counts_good[i] + counts_multi[i] + counts_other[i])
        total_errs = np.sqrt(totals[i])

        if totals[i] > 0.:
            multi_rates[i] += ((float(counts_multi[i]))/float(totals[i]))
            good_rates[i] += ((float(counts_good[i]))/float(totals[i]))
            #rate_errs[i] = rates[i]*np.sqrt((((multi_errs)**2)/((float(counts_multi[i]))**2)) + ((total_errs**2)/((totals[i])**2)))
            multi_test_rate_errs[i] = np.sqrt((multi_rates[i]*(1-multi_rates[i]))/totals[i])
            good_test_rate_errs[i] = np.sqrt((good_rates[i]*(1-good_rates[i]))/totals[i])

    plt.xlabel(r'Galactic Latitude (degrees)',fontsize=18)
    plt.xlim([-90., 90.])
    plt.xticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.ylabel(r'Subjects',fontsize=18)
    plt.legend(loc = "upper left")

    plt.savefig(directory+"/classification_data_raw_numbers_"+strkey+".png")
    plt.savefig(directory+"/classification_data_raw_numbers_"+strkey+".pdf")
  
    plt.figure()
    plt.hist(goodlat, bins=binvec_lat, label=('Good'))

    plt.xlabel("Galactic latitude (degrees)",fontsize=18)
    plt.xlim([-90., 90.])
    plt.xticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.ylabel("Number of subjects", fontsize=18)
    plt.legend(loc = "upper left")

    plt.savefig(directory+"/classification_data_raw_numbers_"+strkey+"_good.png")
    plt.savefig(directory+"/classification_data_raw_numbers_"+strkey+"_good.pdf")
     

    print "Number Histogram Plotted"
    return good_rates, good_test_rate_errs, multi_rates, multi_test_rate_errs


def coord_density_histogram(ratevec, rate_errvec, directory, test, strkey, label):
    binvec_lat = np.arange(-90., 91., 5.)

    hist_x = []
    hist_y = []
    error_x = []
    error_y = []
 
    for i in range(0,(len(binvec_lat.tolist())-1),1):
        hist_x.append(binvec_lat[i])
        hist_x.append(binvec_lat[i+1])
        error_x.append(0.5*(binvec_lat[i] + binvec_lat[i+1]))

    for val in ratevec:
        hist_y.append(val)
        hist_y.append(val)
        error_y.append(val)

    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}

    plt.figure()
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams.update(params)

    plt.gcf().subplots_adjust(left=0.12)
    plt.gcf().subplots_adjust(right=0.95)
    plt.gcf().subplots_adjust(top=0.95)
    plt.gcf().subplots_adjust(bottom=0.12)
    
    plt.plot(hist_x, hist_y)
    plt.errorbar(error_x, error_y, yerr=rate_errvec, fmt='b.')
    plt.xlabel("Galactic latitude (degrees)",fontsize=18)
    plt.xlim([-90., 90.])
    plt.xticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.ylabel(label+" Fraction",fontsize=18)

    #rate

    if test:
        plt.savefig(directory+"/classification_data_rates_test_"+strkey+"_"+label+".png")
        plt.savefig(directory+"/classification_data_rates_test_"+strkey+"_"+label+".pdf")
    else:
        plt.savefig(directory+"/classification_data_rates_"+strkey+"_"+label+".png")
        plt.savefig(directory+"/classification_data_rates_"+strkey+"_"+label+".pdf")
    #print "Histogram saved"
    return "Histogram saved"

def coord_density_histogram_mult(goodratevec, goodrate_errvec, multratevec, multrate_errvec, directory, test, strkey):
    binvec_lat = np.arange(-90., 91., 5.)

    hist_x = []
    hist_y_good = []
    hist_y_mult = []
    error_x = []
    error_y_good = []
    error_y_mult = []
 
    for i in range(0,(len(binvec_lat.tolist())-1),1):
        hist_x.append(binvec_lat[i])
        hist_x.append(binvec_lat[i+1])
        error_x.append(0.5*(binvec_lat[i] + binvec_lat[i+1]))

    for val in goodratevec:
        hist_y_good.append(val)
        hist_y_good.append(val)
        error_y_good.append(val)


    for val in multratevec:
        hist_y_mult.append(val)
        hist_y_mult.append(val)
        error_y_mult.append(val)


    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure()
    plt.plot(hist_x, hist_y_good, label="Good")
    plt.errorbar(error_x, error_y_good, yerr=goodrate_errvec, fmt='b.')
    plt.plot(hist_x, hist_y_mult, 'g', label="Multi")
    plt.errorbar(error_x, error_y_mult, yerr=multrate_errvec, fmt='g.')
    plt.xlabel("Galactic latitude (degrees)",fontsize=18)
    plt.legend(loc="upper right")
    plt.xlim([-90., 90.])
    plt.xticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.ylabel("Fraction")

    if test:
        plt.savefig(directory+"/classification_data_rates_both_test_"+strkey+".png")
        #plt.savefig(directory+"/classification_data_rates_both_test_"+strkey+"_"+label+".pdf")
    else:
        plt.savefig(directory+"/classification_data_rates_"+strkey+"_"+label+".png")
        plt.savefig(directory+"/classification_data_rates_"+strkey+"_"+label+".pdf")
    #print "Histogram saved"
    return "Histogram saved"



def lat_long_heatmap(goodlist, multilist, otherlist, directory, filekey):
    good_lat = []
    good_long = []

    multi_lat = []
    multi_long = []

    other_lat = []
    other_long = []

    total_lat = []
    total_long = []

    for item in goodlist:
        total_lat.append(item.gal_coords[1])
        total_long.append(item.gal_coords[0])

        good_lat.append(item.gal_coords[1])
        good_long.append(item.gal_coords[0])

    for item in multilist:
        total_lat.append(item.gal_coords[1])
        total_long.append(item.gal_coords[0])

        multi_lat.append(item.gal_coords[1])
        multi_long.append(item.gal_coords[0])

    for item in otherlist:
        total_lat.append(item.gal_coords[1])
        total_long.append(item.gal_coords[0])

        other_lat.append(item.gal_coords[1])
        other_long.append(item.gal_coords[0])

    binvec_long = np.arange(0., 361., 5.)
    binvec_lat = np.arange(-90., 91., 5.)

#    long_num = binvec_long.size
#    lat_num = binvec_lat.size

#    long_coords = np.zeros(long_num-1)
#    lat_coords = np.zeros(lat_num-1)

#    for i in range(long_coords.size):
#        long_coords[i] = 0.5*(binvec_long[i] + binvec_long[i+1])

#    for i in range(lat_coords.size):
#        lat_coords[i] = 0.5*(binvec_lat[i] + binvec_lat[i+1])

    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}

    plt.figure()
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams.update(params)

    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(right=0.95)
    plt.gcf().subplots_adjust(top=0.95)
    plt.gcf().subplots_adjust(bottom=0.12)

    total_counts, total_xedges, total_yedges, image_total = plt.hist2d(total_long, total_lat, bins=[binvec_long,binvec_lat],norm=mpl.colors.LogNorm())
    #total_array = image_total.get_array()
    #total_verts = image_total.get_offsets()
    plt.xlim([0., 360.])
    plt.ylim([-90., 90.])
    plt.xticks([0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330., 360.])
    plt.yticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.colorbar()
    plt.savefig(directory+'/'+filekey+'_total_colormap.pdf')

    plt.figure()
    multi_counts, multi_xedges, multi_yedges, image_multi = plt.hist2d(multi_long, multi_lat, bins=[binvec_long,binvec_lat],norm=mpl.colors.LogNorm())
    #multi_array = image_multi.get_array()
    plt.xlim([0., 360.])
    plt.ylim([-90., 90.])
    plt.xticks([0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330., 360.])
    plt.yticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.colorbar()
    plt.savefig(directory+'/'+filekey+'_multi_colormap.pdf')

    plt.figure()
    good_counts, good_xedges, good_yedges, image_good = plt.hist2d(good_long, good_lat, bins=[binvec_long, binvec_lat], norm=mpl.colors.LogNorm())
    plt.xlim([0.,360.])
    plt.ylim([-90., 90.])
    plt.xticks([0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330., 360.])
    plt.yticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.colorbar()
    plt.savefig(directory+'/'+filekey+'_good_colormap.pdf')

    print total_counts.shape
    print multi_counts.shape
    print good_counts.shape

    multi_ratio_array = np.zeros(multi_counts.shape)
    good_ratio_array = np.zeros(good_counts.shape)

    for i in range(72):
        for j in range(36):
            multi_ratio_array[i][j] = multi_counts[i][j] / total_counts[i][j]
            good_ratio_array[i][j] = good_counts[i][j] / total_counts[i][j]

    x2,y2 = np.meshgrid(binvec_long, binvec_lat)

    print (x2.shape, y2.shape)

    print multi_ratio_array.shape
    print np.swapaxes(multi_ratio_array,0,1).shape

    print good_ratio_array.shape
    print np.swapaxes(good_ratio_array,0,1,).shape

    minplane = [-5., -5.]
    maxplane = [5., 5.]

    plt.figure()
    plt.pcolormesh(x2, y2, np.swapaxes(multi_ratio_array,0,1), vmin = 0., vmax = 1.)
    plt.xlim([0., 360.])
    plt.ylim([-90., 90.])
    plt.xticks([0., 60., 120., 180., 240., 300., 360.])
    plt.yticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.colorbar()
    plt.xlabel('Galactic longitude', fontsize=18)
    plt.ylabel('Galactic latitude', fontsize=18)

    plt.plot([0., 360.], minplane, 'w--', linewidth=4)
    plt.plot([0., 360.], maxplane, 'w--', linewidth=4)

    plt.plot([280.4652, 302.8084], [-32.8884, -44.3277], 'wx', markersize=18, mew=5)

    plt.savefig(directory+'/multi_density_colormap'+filekey+'.pdf')
    plt.savefig(directory+'/multi_density_colormap'+filekey+'.png')
    plt.close()

    plt.figure()
    plt.pcolormesh(x2, y2, np.swapaxes(good_ratio_array,0,1), vmin = 0., vmax = 1.)
    plt.xlim([0., 360.])
    plt.ylim([-90., 90.])
    plt.xticks([0., 60., 120., 180., 240., 300., 360.])
    plt.yticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.colorbar()
    plt.xlabel('Galactic longitude', fontsize=18)
    plt.ylabel('Galactic latitude', fontsize=18)

    plt.plot([0., 360.], minplane, 'w--', linewidth=4)
    plt.plot([0., 360.], maxplane, 'w--', linewidth=4)

    plt.plot([280.4652, 302.8084], [-32.8884, -44.3277], 'wx', markersize=18, mew=5)

    plt.savefig(directory+'/good_density_colormap'+filekey+'.pdf')
    plt.savefig(directory+'/good_density_colormap'+filekey+'.png')
    plt.close()

    np.set_printoptions(threshold=np.inf)
    #print np.swapaxes(good_ratio_array,0,1)
    #print good_counts
    #print total_counts

    np.savetxt(directory+'/good_ratio_array_swap.txt',np.swapaxes(good_ratio_array,0,1),fmt='%1.4e',newline='\n')
    np.savetxt(directory+'/good_counts.txt',good_counts,fmt='%1.4e',newline='\n')
    np.savetxt(directory+'/total_counts.txt',total_counts,fmt='%1.4e',newline='\n')
    

    #f1 = open(directory+'/raw_numbers_binned_'+strkey+'.dat','w')
    #for i in range(0, len(binvec_lat)-1, 1):
    #    f1.write(str(counts_good[i])+','+str(counts_multi[i])+','+str(counts_other[i])+'\n')
    #f1.close()

    doneness1 = "Done"

    print filekey, "Done"

    return doneness1


def lat_long_remaining(remaining_list,directory):
    remaining_lats = []
    remaining_longs = []

    for item in remaining_list:
        remaining_lats.append(item.gal_coords[1])
        remaining_longs.append(item.gal_coords[0])

    binvec_long = np.arange(0., 361., 5.)
    binvec_lat = np.arange(-90., 91., 5.)

    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}

    plt.figure()
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams.update(params)

    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(right=0.95)
    plt.gcf().subplots_adjust(top=0.95)
    plt.gcf().subplots_adjust(bottom=0.12)

    total_counts, total_xedges, total_yedges, image_total = plt.hist2d(remaining_longs, remaining_lats, bins=[binvec_long,binvec_lat],norm=mpl.colors.LogNorm())
    #total_array = image_total.get_array()
    #total_verts = image_total.get_offsets()
    plt.xlim([0., 360.])
    plt.ylim([-90., 90.])
    plt.xticks([0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330., 360.])
    plt.yticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.colorbar()
    plt.savefig('remaining_colormap.pdf')

    np.savetxt(directory+'/remaining_totals.txt',total_counts,fmt='%1.4e',newline='\n')

    return "Done"


def bright_number_histogram(goodlist, multilist, otherlist, directory, brightlim):
    #build lists
    goodJmag = []
    #goodlong = []
    for item in goodlist:
        goodJmag.append(item.jmag)
        #goodlat.append(item.gal_coords[1])

    multiJmag = []
    #goodlong = []
    for item in multilist:
        multiJmag.append(item.jmag)
        #goodlat.append(item.gal_coords[1])

    otherJmag = []
    #goodlong = []
    for item in otherlist:
        otherJmag.append(item.jmag)
        #goodlat.append(item.gal_coords[1])
    
    binvec_Jmag = np.arange(brightlim, 14.6, 0.5)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure()
    plt.hist([goodJmag, multiJmag, otherJmag], bins=binvec_Jmag, stacked=True, label=('Good', 'Multi', 'Other'))
    counts_good,edges_good = np.histogram(goodJmag, bins=binvec_Jmag)
    counts_multi,edges_multi = np.histogram(multiJmag, bins=binvec_Jmag)
    counts_other,edges_other = np.histogram(otherJmag, bins=binvec_Jmag)

    f1 = open(directory+'/raw_numbers_binned_brightness.dat','w')
    for i in range(0, len(binvec_Jmag)-1, 1):
        f1.write(str(counts_good[i])+','+str(counts_multi[i])+','+str(counts_other[i])+'\n')
    f1.close()

    totals = np.zeros(len(binvec_Jmag) - 1)
    rates = np.zeros(len(binvec_Jmag) - 1)
    rate_errs = np.zeros(len(binvec_Jmag) - 1)
    test_rate_errs = np.zeros(len(binvec_Jmag) - 1)
    for i in range(0,len(binvec_Jmag)-1,1):
        multi_errs = np.sqrt(float(counts_multi[i]))
        totals[i] += float(counts_good[i] + counts_multi[i] + counts_other[i])
        total_errs = np.sqrt(totals[i])

        if totals[i] > 0.:
            rates[i] += ((float(counts_multi[i]))/float(totals[i]))
            rate_errs[i] = rates[i]*np.sqrt((((multi_errs)**2)/((float(counts_multi[i]))**2)) + ((total_errs**2)/((totals[i])**2)))
            test_rate_errs[i] = np.sqrt((rates[i]*(1-rates[i]))/totals[i])

    plt.xlabel("2MASS J magnitude")
    plt.xlim([brightlim, 14.5])
    #plt.ylim([0.,1.])
    plt.xticks(binvec_Jmag)
    plt.ylabel("Number of subjects")
    plt.legend(loc = "upper right")

    plt.savefig(directory+"/classification_data_raw_numbers_brightness.png")
    plt.savefig(directory+"/classification_data-raw_numbers_brightness.pdf")

    print "Number Brightness Histogram Plotted"
    return rates, rate_errs, test_rate_errs


def get_background_galaxies(classified_good_vec):
    #classified_good_count = 0.
    #total = float(len(subjects))
    expected_good_after_background_galaxies = 0.

    #classified_good_vec = []

    #for subject in subjects:
    #    if subject.majgood:
    #        classified_good_vec.append(subject)

    classified_good_count = float(len(classified_good_vec))

    log_flux_density_vec = np.array([-1.550, -1.400, -1.250, -1.10, -0.950, -0.800, -0.650, -0.500, -0.350, -0.200, -0.050, 0.100, 0.250, 0.400, 0.550, 0.700, 0.850, 1.000, 1.150, 1.300, 1.450, 1.600])
    flux_density_vec = 10.**(log_flux_density_vec)

    cumul_counts_vec = np.array([2.1e8, 1.2e8, 9.6e7, 6.1e7, 3.9e7, 2.9e7, 1.8e7, 8.6e6, 4.4e6, 2.3e6, 1.2e6, 7.1e5, 4.1e5, 2.3e5, 1.3e5, 7.7e4, 5.2e4, 3.4e4, 2.0e4, 1.3e4, 7.3e3, 4.4e3])

    #steradians_per_square_degree = (np.pi**2)/(180.**2)

    #w4_beam_square_degrees = np.pi*((12./3600.)**2)

    #w4_beam_steradians = w4_beam_square_degrees * steradians_per_square_degree

    w4_beam_steradians = 2*np.pi*(1-np.cos(np.pi/54000.))

    number_contaminated = 0.

    for good_subject in classified_good_vec:
        w4mag = good_subject.w4mag
        Hmag = good_subject.hmag

        f_Hmag = 1024.*10.**(-0.4*Hmag)   

        extrap_f_w4 = f_Hmag * (22./1.662)**2.

        f_w4 = 8.363 * 10.**(-0.4*w4mag)

        extrap_f_w4_mjy = extrap_f_w4 * 1000.

        remaining_excess = f_w4 - extrap_f_w4

        remaining_w4 = -2.5*np.log10(remaining_excess/8.363)

        if good_subject.w1mag - remaining_w4 > 0.25:
            number_contaminated += 1
        else:
            contamination_level_mag = w4mag + (-2.5*np.log10((10**0.1) - 1.))

            contamination_level_flux_janskys = 8.363*(10.**(-contamination_level_mag/2.5))

            contamination_level_flux_millijanskys = contamination_level_flux_janskys * 1000.

            contaminants_per_steradian = np.interp(contamination_level_flux_millijanskys, flux_density_vec, cumul_counts_vec)

            cut_off_top_contaminants_per_steradian = np.interp(extrap_f_w4_mjy, flux_density_vec, cumul_counts_vec)

            contamination_fraction = (contaminants_per_steradian * w4_beam_steradians) - (cut_off_top_contaminants_per_steradian * w4_beam_steradians)

            if contamination_fraction >= 1.:
                number_contaminated += 1
            else: 
                number_contaminated += contamination_fraction

    total_no_galaxies = classified_good_count - number_contaminated

    #no_galaxies_fraction = total_no_galaxies/total

    return total_no_galaxies

def bright_density_histogram(ratevec, rate_errvec, directory, test, brightlim):
    binvec_Jmag = np.arange(brightlim, 14.6, 0.5)

    hist_x = []
    hist_y = []
    error_x = []
    error_y = []
 
    for i in range(0,(len(binvec_Jmag.tolist())-1),1):
        hist_x.append(binvec_Jmag[i])
        hist_x.append(binvec_Jmag[i+1])
        error_x.append(0.5*(binvec_Jmag[i] + binvec_Jmag[i+1]))
        print binvec_Jmag[i],binvec_Jmag[i+1],ratevec[i]

    for val in ratevec:
        hist_y.append(val)
        hist_y.append(val)
        error_y.append(val)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure()
    plt.plot(hist_x, hist_y)
    plt.errorbar(error_x, error_y, yerr=rate_errvec, fmt='b.')
    plt.plot(binvec_Jmag, 29.1216*(binvec_Jmag**(-1.55)))
    plt.xlabel("2MASS J magnitude")
    plt.xlim([brightlim, 14.5])
    plt.ylim([0.,1.])
    plt.xticks(binvec_Jmag)
    plt.ylabel("Multiple fraction")
    plt.legend(loc = "upper right")

    if test:
        plt.savefig(directory+"/classification_data_brightness_rates_test.pdf")
        plt.savefig(directory+"/classification_data_brightness_rates_test.png")
    else:
        plt.savefig(directory+"/classification_data_brightness_rates.pdf")
        plt.savefig(directory+"/classification_data_brightness_rates.png")
    #print "Histogram saved"
    return "Histogram saved"

#def bright_density_latitude_plots(goodlist, multilist, otherlist, directory, brightlim):
    

def bright_density_latlong(goodlist, multilist, otherlist, directory,brightlim):
    binvec_Jmag = np.arange(brightlim, 14.6, 0.5)
    bin_name_list = []
    
    for i in range(binvec_Jmag.size-1):
        bin_name_list.append((binvec_Jmag[i],binvec_Jmag[i+1]))

    bin_dict_good = {k: [] for k in bin_name_list}
    bin_dict_multi = {k: [] for k in bin_name_list}
    bin_dict_other = {k: [] for k in bin_name_list}

    for obj in goodlist:
        for key in bin_dict_good.keys():
            if obj.jmag > key[0] and obj.jmag < key[1]:
                bin_dict_good[key].append(obj)

    for obj in multilist:
        for key in bin_dict_multi.keys():
            if obj.jmag > key[0] and obj.jmag < key[1]:
                bin_dict_multi[key].append(obj)

    for obj in otherlist:
        for key in bin_dict_other.keys():
            if obj.jmag > key[0] and obj.jmag < key[1]:
                bin_dict_other[key].append(obj)
                

    for key in bin_name_list:
        goodlist_use = bin_dict_good[key]
        multilist_use = bin_dict_multi[key]
        otherlist_use = bin_dict_other[key]
        strkey = str(key[0])+'_'+str(key[1])+'_'

        good_rates, good_test_rate_errs, multi_rates, multi_test_rate_errs = coord_number_histogram(goodlist_use, multilist_use, otherlist_use, directory, strkey)

        #density_histogram_status = coord_density_histogram(rates, rate_errs, directory, False, 'full_data_set')
        good_density_histogram_status_test = coord_density_histogram(good_rates, good_test_rate_errs, directory, True, strkey, 'good')
        multi_density_histogram_status_test = coord_density_histogram(multi_rates, multi_test_rate_errs, directory, True, strkey,'multi')


        lat_long_heatmap_status = lat_long_heatmap(goodlist_use, multilist_use, otherlist_use, directory,strkey)

    return "Jmag binned latitude plots done"


def bright_density_latbright(goodlist, multilist, otherlist, directory,brightlim):
    binvec_Jmag = np.arange(brightlim, 14.6, 0.5)
    bin_name_list = []

    good_lat = []
    good_bright = []

    multi_lat = []
    multi_bright = []

    other_lat = []
    other_bright = []

    total_lat = []
    total_bright = []

    for item in goodlist:
        total_lat.append(item.gal_coords[1])
        total_bright.append(item.jmag)

        good_lat.append(item.gal_coords[1])
        good_bright.append(item.jmag)

    for item in multilist:
        total_lat.append(item.gal_coords[1])
        total_bright.append(item.jmag)

        multi_lat.append(item.gal_coords[1])
        multi_bright.append(item.jmag)

    for item in otherlist:
        total_lat.append(item.gal_coords[1])
        total_bright.append(item.jmag)

        multi_lat.append(item.gal_coords[1])
        multi_bright.append(item.jmag)


    #binvec_bright = np.arange(0., 361., 5.)
    binvec_lat = np.arange(-90., 91., 5.)

#    long_num = binvec_long.size
#    lat_num = binvec_lat.size

#    long_coords = np.zeros(long_num-1)
#    lat_coords = np.zeros(lat_num-1)

#    for i in range(long_coords.size):
#        long_coords[i] = 0.5*(binvec_long[i] + binvec_long[i+1])

#    for i in range(lat_coords.size):
#        lat_coords[i] = 0.5*(binvec_lat[i] + binvec_lat[i+1])

    plt.figure()
    total_counts, total_xedges, total_yedges, image_total = plt.hist2d(total_bright, total_lat, bins=[binvec_Jmag,binvec_lat],norm=mpl.colors.LogNorm())
    #total_array = image_total.get_array()
    #total_verts = image_total.get_offsets()
    plt.xlim([brightlim, 14.5])
    plt.ylim([-90., 90.])
    plt.xticks(binvec_Jmag.tolist())
    plt.yticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.colorbar()
    plt.savefig(directory+'/latbright_total_colormap.pdf')

    plt.figure()
    multi_counts, multi_xedges, multi_yedges, image_multi = plt.hist2d(multi_bright, multi_lat, bins=[binvec_Jmag,binvec_lat],norm=mpl.colors.LogNorm())
    #multi_array = image_multi.get_array()
    plt.xlim([brightlim, 14.5])
    plt.ylim([-90., 90.])
    plt.xticks(binvec_Jmag.tolist())
    plt.yticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.colorbar()
    plt.savefig(directory+'/latbright__multi_colormap.pdf')

    print total_counts.shape
    print multi_counts.shape

    ratio_array = np.zeros(multi_counts.shape)

    for i in range(binvec_Jmag.size-1):
        for j in range(36):
            ratio_array[i][j] = multi_counts[i][j] / total_counts[i][j]

    x2,y2 = np.meshgrid(binvec_Jmag, binvec_lat)

    print (x2.shape, y2.shape)

    print ratio_array.shape
    print np.swapaxes(ratio_array,0,1).shape

    minplane = [-5., -5.]
    maxplane = [5., 5.]

    plt.figure()
    plt.pcolormesh(x2, y2, np.swapaxes(ratio_array,0,1), vmin = 0., vmax = 1.)
    plt.xlim([brightlim, 14.5])
    plt.ylim([-90., 90.])
    plt.xticks(binvec_Jmag.tolist())
    plt.yticks([-90., -60., -30., 0., 30., 60., 90.])
    plt.colorbar()
    plt.xlabel('J magnitude', fontsize=16)
    plt.ylabel('Galactic latitude', fontsize=16)

    plt.plot([brightlim, 14.5], minplane, 'w--', linewidth=2)
    plt.plot([brightlim, 14.5], maxplane, 'w--', linewidth=2)

    #plt.plot([280.4652, 302.8084], [-32.8884, -44.3277], 'wx')

    plt.savefig(directory+'/density_lat_bright_colormap.pdf')
    plt.close()

    doneness1 = "Done"

    #print filekey, "Done"

    return doneness1
    
