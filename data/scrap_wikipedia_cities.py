#!/usr/bin/env python2
# coding: utf-8

import os
import pandas as pd
import urllib2

from bs4 import BeautifulSoup

wiki_alpha_3 = {
    'name': 'wiki_alpha_3',
    'url': 'https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3',
}

wiki_cities_lat = {
    'name': 'wiki_cities_lat',
    'url': 'https://en.wikipedia.org/wiki/List_of_cities_by_latitude',
}

features = [wiki_alpha_3, wiki_cities_lat]

# Read Wikipedia articles

for feature in features:
    filename = os.path.join(
        os.path.dirname(__file__), feature['name'] + '.html')
    
    if not os.path.isfile(filename):
        print filename, 'not available. Downloading...'
        req = urllib2.Request(feature['url'])
        res = urllib2.urlopen(req)

        with open(filename, 'w') as f:
            f.write(res.read())

    with open(filename, 'r') as f:
        feature['html'] = f.read()

# Parse Alpha 3 article

a3_soup = BeautifulSoup(wiki_alpha_3['html'], 'html')

alpha3_name = dict()
countries = dict()

for column in a3_soup.find_all('table', cellspacing='0'):
    trs = column.find_all('tr')
    for tr in trs:
        tds = tr.find_all('td')
        if tds and len(tds) > 1:
            alpha3 = tds[0].get_text()
            name = tds[1].get_text()
            alpha3_name[alpha3] = name
            countries[name] = {'alpha3': alpha3}

# Parse list of cities article

cities_soup = BeautifulSoup(wiki_cities_lat['html'], 'html')

countries_props = dict()

for table in cities_soup.find_all('table', class_='wikitable'):
    trs = table.find_all('tr')
    for tr in trs:
        tds = tr.find_all('td')
        values = [td.get_text() for td in tds]
        if len(values) == 5:
            country_name = values[4]
            # Remove country flag artifact
            country_name = country_name.replace(u'\xa0','') 
            countries_props[country_name] = {
                'latitude':     values[0],
                'longitude':    values[1],
                'city':         values[2],
                'province':     values[3]
            }

# Agregate

for country in countries.keys():

    for prop in ['latitude', 'longitude', 'city', 'province']:
        countries[country][prop] = countries_props.get(country, {}).get(prop, None)

    for dim in ['latitude', 'longitude']:

        dimension = countries[country][dim]
        if not dimension:
            continue

        # Convert degrees to decimal

        degree, side = dimension.split(u'′')
        D, M = degree.split(u'°')
        D, M = float(D), float(M)

        sign = 1
        if side in ['S', 'W']:
            sign = -1

        countries[country][dim + '_dec'] = sign * (D + M/60.)

for country in countries.keys():
    if len([prop for prop in countries[country].values() if prop != None]) == 1:
        pass #print country

countries_df = pd.DataFrame(countries)
