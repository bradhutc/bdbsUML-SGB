import pandas as pd
from astroquery.simbad import Simbad
from astropy import coordinates as coord
from astropy import units as u

csv_file_path = '/N/project/catypGC/BDBS/bdbsparallaxprocessed_data.csv'
df = pd.read_csv(csv_file_path).sample(5)

# Initialize Simbad query
simbad_query = Simbad()
simbad_query.add_votable_fields('otype', 'flux(J)', 'flux(H)', 'flux(K)')

# Create an empty list to store matched stars
matched_stars = []

# Loop through each star
for index, row in df.iterrows():
    ra = row['ra']
    dec = row['dec']
    umag = row['umag']
    gmag = row['gmag']
    rmag = row['rmag']
    imag = row['imag']
    zmag = row['zmag']
    ymag = row['ymag']

    # Query Simbad for each star
    result_table = simbad_query.query_region(
        coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'),
        radius='0d0m01s'
    )

    # Check if there are any results
    if result_table is not None and len(result_table) > 0:
        # Convert Astropy Table to Pandas DataFrame
        result_df = result_table.to_pandas()

        star_name = result_df['MAIN_ID'].values[0]
        object_type = result_df['OTYPE'].values[0]
        ra_prec = result_df['RA_PREC'].values[0]
        dec_prec = result_df['DEC_PREC'].values[0]
        jmag = result_df['FLUX_J'].values[0] 
        hmag = result_df['FLUX_H'].values[0] 
        kmag = result_df['FLUX_K'].values[0] 

        # Add the matched star to the list
        matched_stars.append((ra, dec, ra_prec, dec_prec, umag, gmag, rmag, imag, zmag, ymag, star_name, object_type, jmag, hmag, kmag))

if matched_stars:
    print("\nMatch found for the following coordinates:")
    for ra, dec, ra_prec, dec_prec, umag, gmag, rmag, imag, zmag, ymag, star_name, object_type, jmag, hmag, kmag in matched_stars:
        print(f"RA: {ra}, Dec: {dec}, RA_Prec: {ra_prec}, Dec_Prec: {dec_prec}, umag: {umag}, gmag: {gmag}, rmag: {rmag}, imag: {imag}, zmag: {zmag}, ymag: {ymag}, Star Name: {star_name}, Object Type: {object_type}, J: {jmag}, H: {hmag}, K: {kmag}")

    matched_stars_df = pd.DataFrame(matched_stars, columns=['ra', 'dec', 'ra_prec', 'dec_prec', 'umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag', 'star_name', 'object_type', 'Jmag', 'Hmag', 'Kmag'])
    matched_stars_df.to_csv('simbad_matched.csv', index=False)
    print("Matched stars saved to 'simbad_matched.csv'.")

else:
    print("No Simbad results found for the given stars.")
