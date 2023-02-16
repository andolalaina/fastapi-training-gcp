def validateGeojson(geojson):
    if not geojson.type:
        return False
    if not geojson.features:
        return False
    return True