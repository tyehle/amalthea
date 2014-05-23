import psycopg2

db = "zipcodes"
host = "163.118.78.22"
user = "swhite"
port = 5432
conn = psycopg2.connect("dbname={} host={} user={}".format(db,host,user))
cur = conn.cursor()

def get_zip(lat,lon):
    cur.execute("SELECT zcta5ce10 from tl_2013_us_zcta510 where ST_Intersects('POINT({lon} {lat})'::geometry, geom) ;".format(lat=lat,lon=lon))
    data = cur.fetchall()
    if len(data) == 0:
        return 0
    else:
        return data[0][0]

# a test to make sure the db is set up
# print get_zip(-122.3775,37.827)
