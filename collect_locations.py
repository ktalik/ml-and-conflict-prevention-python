#!/usr/bin/env python2
# -*- coding: utf8 -*-

import csv
import geopy
import time

import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from datetime import datetime

longitudes, latitudes = list(), list()

##
# Plot World
##
plot = True

if plot:
    # miller projection
    bmap = Basemap(projection='mill')
    bmap.drawcoastlines()
    bmap.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
    bmap.drawmeridians(np.arange(bmap.lonmin,bmap.lonmax+30,60),labels=[0,0,0,1])
    bmap.drawmapboundary(fill_color='aqua')
    bmap.fillcontinents(color='coral', lake_color='aqua')

    # shade the night areas, with alpha transparency so the
    # bmap shows through. Use current time in UTC.
    date = datetime.utcnow()
    CS=bmap.nightshade(date)

    plt.title('Day/Night Map for %s (UTC)' % date.strftime("%d %b %Y %H:%M:%S"))
    plt.ion()
    plt.show()

##
# Get data
##

locator = geopy.geocoders.Nominatim()

with open('data/districts.csv', 'rU') as districts_file, \
     open('output/locations.csv', 'w') as locations_file:
    reader = csv.reader(districts_file)
    writer = csv.writer(locations_file)

    reader = list(reader)
    num_dist = len(reader)

    # Consider every "NAME" column
    for line_num, row in enumerate(reader):
        location_data = [row[16], row[13], row[9], row[5], row[3]]

        # Consider most detailed location string as possible
        for i in range(len(location_data)):
            location_string = reduce(lambda a, b: a + ', ' + b, location_data[i:])

            try:
                location = locator.geocode(location_string)
            except Exception as error:
                print location_string, '- exception:', error
                continue

            if location:
                longitudes.append(location.longitude)
                latitudes.append(location.latitude)
                print location_string, '- location OK'
                print 'Done', line_num, 'out of', num_dist, \
                    '({}%)'.format(100.*float(line_num)/float(num_dist))

                # Save result
                # OBJECTID, ID_0, ISO, NAME_0, LON, LAT
                writer.writerow([
                    row[0], row[1], row[2], row[3],
                    location.longitude, location.latitude])

                if plot:
                    # Scatter plot locations
                    x, y = bmap(longitudes, latitudes)
                    bmap.scatter(x, y, 30, marker='o', color='k', zorder=2)
                    plt.pause(0.001)

                break
            else:
                print location_string, '- invalid location'





