import pymongo
import reverse_geo
import csv
from dateutil import parser

client = pymongo.MongoClient('163.118.78.22', 27017)
db = client['crimes']
crimes = db.crimes
years = ['2007', '2008', '2009', '2010']
months = range(1,13)
chunk_size = 1000

# iterate over the months and years to add to the db
for y in years:
    for m in months:
        # grab the file we need
        filename = '{0}-{1}.csv'.format(y, m)
        data = csv.DictReader(open(filename,'r'))
        print('dumping data in {0}'.format(filename))
        data_chunk = []
        chunks = 0
        # iterate through the records in the file
        for d in data:
            # add the zipcode field
            if d['Latitude'] != 0 and d['Longitude'] != 0:
                d['zipcode'] = reverse_geo.get_zip(d['Latitude'], d['Longitude'])
            del d['Link'] # we don't care about the link
            lower = dict(zip([k.lower() for k in d.keys()],d.values()))
            lower['date'] = parser.parse(lower['date'])
            data_chunk.append(lower)
            # if there are chunk_size elements in the list of records add them all to the db
            if len(data_chunk) >= chunk_size:
                crimes.insert(data_chunk)
                chunks = chunks + 1
                print(chunks)
                data_chunk = [] # reset the list of records

        # insert any leftover data
        crimes.insert(data_chunk)
