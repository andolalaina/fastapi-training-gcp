import json
import ee
import requests
from fastapi import APIRouter
from src.services.applicatif.classificationApp import ClassificationApp

router = APIRouter(
    prefix="/classifications",
    tags=["classifications"],
    responses={404: {"description": "Not found"}}
)

classificationSvc = ClassificationApp()

@router.get("/")
async def classify_geojson() -> object:
    roi = ee.Geometry.Polygon(
        [[[44.81333765501404, -17.2411250284625],
          [45.25279078001404, -20.854292143915252],
          [46.83482203001404, -21.346276455964947],
          [48.41685328001404, -20.44307050853734],
          [49.25181421751404, -17.157164637214876],
          [49.38365015501404, -15.681853091000558],
          [47.05454859251404, -15.554885053962325],
          [45.99986109251404, -16.146722815776897]]])
    ground_truth = ee.FeatureCollection("projects/ee-michael-bpo/assets/BD_JECAM_CIRAD_2021_dec")
    mada_crop = ground_truth.filter(ee.Filter.eq('Country','Madagascar')) \
          .filterBounds(roi)
    def create_year (feature) :
        return feature.set({'Year':ee.Date(feature.get('AcquiDate')).get('year')})
    mada_crop_year = mada_crop.map(create_year)
    def create_month (feature) :
        return feature.set({'Month':ee.Date(feature.get('AcquiDate')).get('month')})
    mada_crop_fc = mada_crop_year.map(create_month)
    feat_crop = mada_crop_fc.filter(ee.Filter.eq("Year",2019)) \
               .filter(ee.Filter.Or(ee.Filter.eq("Month",1), ee.Filter.eq("Month",2))) \
               .filter(ee.Filter.neq("CropType1",""))
    mapping = ee.Dictionary({
    'Agricultural bare soil': 'Others',
    'Apple tree': 'Others',
    'Asparagus': 'Others',
    'Bean':'Others',
    'Cabbage': 'Others',
    'Carrot': 'Others',
    'Cash woody crop': 'Others',
    'Cassava': 'Others',
    'Cucumber': 'Others',
    'Fruit crop': 'Others',
    'Fruit-bearing vegetable': 'Others',
    'Grasses and other fodder crop': 'Others',
    'Leafy or stem vegetable': 'Others',
    'Leguminous': 'Others',
    'Maize': 'Maize',
    'Market gardening': 'Others',
    'Oilseed crop': 'Others',
    'Pea': 'Others',
    'Peach tree': 'Others',
    'Pear tree': 'Others',
    'Pine': 'Others',
    'Potato': 'Others',
    'Rice': 'Rice',
    'Root, bulb or tuberous vegetable': 'Others',
    'Soybean': 'Others',
    'Sweet potato': 'Others',
    'Taro': 'Others',
    'Tomato': 'Others',
    'Vineyard': 'Others',
    'Young fallow': 'Others'
     })
    def create_crop (feature):
        return feature.set('Crop', mapping.get(feature.get('CropType1')))
    feat_crop = feat_crop.select(['CropType1']).map(create_crop)
    s2Sr = ee.ImageCollection('COPERNICUS/S2_SR')
    START_DATE = ee.Date('2019-01-01')
    END_DATE = ee.Date('2019-03-01')
    s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    MAX_CLOUD_PROBABILITY = 10
    def maskClouds(img):
        clouds = ee.Image(img.get('cloud_mask')).select('probability')
        isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY)
        return img.updateMask(isNotCloud)
    def maskEdges(s2_img):
        return s2_img.updateMask(
            s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask()))
    criteria = ee.Filter.And(
    ee.Filter.bounds(roi), ee.Filter.date(START_DATE, END_DATE))
    s2Sr = s2Sr.filter(criteria).map(maskEdges)
    s2Clouds = s2Clouds.filter(criteria)
    s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply(
        primary = s2Sr,
        secondary = s2Clouds,
        condition = ee.Filter.equals(leftField = 'system:index', rightField = 'system:index')
        )
    image = ee.ImageCollection(s2SrWithCloudMask).map(maskClouds).median()
    def addIndices(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename(['ndvi'])
        ndbi = image.normalizedDifference(['B11', 'B8']).rename(['ndbi'])
        mndwi = image.normalizedDifference(['B3', 'B11']).rename(['mndwi'])
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('ndwi')
        bsi = image.expression(
            '(( X + Y ) - (A + B)) /(( X + Y ) + (A + B)) ', {
            'X': image.select('B11'), #swir1
            'Y': image.select('B4'),  #red
            'A': image.select('B8'), # nir
            'B': image.select('B2'), # blue
        }).rename('bsi')
        return image.addBands(ndvi).addBands(ndbi).addBands(mndwi).addBands(ndwi).addBands(bsi)

    image = addIndices(image)
    mada_crop_encoded = feat_crop.remap(['Maize','Others','Rice'],[0,1,2],'Crop')
    mada_crop_encoded = mada_crop_encoded.randomColumn(
    columnName = 'random',
    seed = 0
    )
    split = 0.7
    training = mada_crop_encoded.filter(ee.Filter.lt('random', split))
    validation = mada_crop_encoded.filter(ee.Filter.gte('random', split))
    training = image.sampleRegions(
        collection = training,
        properties = ['Crop'],
        tileScale = 16,
        scale = 10
        )

    validation = image.sampleRegions(
        collection = validation,
        properties = ['Crop'],
        tileScale = 16,
        scale = 10
        )
    
    classifier = ee.Classifier.smileRandomForest(50).train(
        features = training,
        classProperty = 'Crop',
        inputProperties = ['B2','B3','B4','B8','B11','B12']#['B2','B3','B4','B8','B11','B12','ndvi','ndbi','mndwi','bsi','ndwi']
        )
    trainAccuracy = classifier.confusionMatrix()
    validated = validation.classify(classifier)

    ####################################################
    ####### import mask rcnn result geojson from ny antso
    maskrcnn_output = requests.post("http://10.2.54.212:5005/get_crop_seg", roi.toGeoJSONString()).json()
    print("-------------------- eto ---------------------")
    print(maskrcnn_output)
    ########################
    
    input_class = ee.FeatureCollection(maskrcnn_output)
    bound_geo = input_class.geometry().bounds()

    inf_image = image.select(['B2','B3','B4','B8','B11','B12']).clip(input_class)

    inference = inf_image.classify(classifier)  

    fc_inference = inference.select("classification").reduceToVectors(
        scale = 1,
        geometry = bound_geo,
        geometryType = 'polygon',
        eightConnected = False,
        labelProperty = 'crop',
        )
    # x = json.loads(str().replace("'", "\""))
    return fc_inference.getInfo()
    