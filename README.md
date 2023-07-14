# Motivation
I wanted to use Strava's API to gain access to my activity data, particularly the GPS track information. I wasn't
terribly happy with how Strava rendered my data into a heatmap on their website, so I figured I'd give it a try here.

# Data
The first step was to get retrieve my data from Strava. There are at least two ways to do this, probably more if you're
creative. The first and easiest one is to download your data from their website. The second is to make use of Strava's
API. I decided to opt for the second option to gain access to my data so that I could easily add new activities in the
future without having to download my entire profile again.

Instructions for both of these ideas can be readily found on Strava's website.

I've made my data publicly available in this repo so that any code should just work out of the box. I've also uploaded
the tiles that I used when generating the heatmap below.

# Heatmap
There is plenty of analysis to be done on the data I retrieved, but the first thing I was really jazzed to look at was a
heatmap indicating where I've been on my bike in the areas surrounding Boulder, CO. as I said, I wasn't terribly happy
with the one that Strava produced. Overall, not terribly bright, but probably mostly because I had no control over
any of the features. There are plenty of guides around the web that helped me understand how best to make my own, but I
want to mention one webpage in particular that I spent a solid amount of time reading through to understand how best to
convert from latitude and longitude to map tile coordinates, and how the pixel values need to be modified in turn. Many
of these details can be found on the linked page of the
[OpenStreetMap wiki](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames).

Ultimately, it came down to histogram equalization, kernel density estimation, a gaussian filter, accounting for maximum
pixel accumulation, and then laying that down on top of a map. Choosing and getting the tiles was another consideration.
I really like the look of Wikimedia's tiles, but they don't necessarily support fetching them. Specifying a descriptive
User-Agent in my requests wasn't sufficient to bypass them automatically blocking my requests, so I instead opted to go
for the free tier of [geoapify](https://www.geoapify.com/map-tiles). I used their osm-bright-grey map tiles, and then
manually inverted the color scheme after stitching them together for a dark look to contrast the heat of my tracks.

Laying my tracks on top of these stitched tiles turned out pretty well in my opinion.
<img src="./img/heatmap.png">

# Generation
I've added an example Jupyter notebook that goes through the process of generating one of these heatmaps from end to
end starting from data acquisition to heatmap generation.