// Define the ROI (Region of Interest) using a bounding box
// var bbox = [[138.5206051676438, 36.425420321930716], [ 138.5206051676438, 36.30034394260757], [138.7465881498856, 36.30034394260757], [138.7465881498856, 36.425420321930716]];  // Replace with actual coordinates

// var roi = ee.Geometry.Polygon(bbox);
// Define the ROI using ee.Geometry.Polygon
var roi = ee.Geometry.Polygon([
  [
    [138.5206051676438, 36.425420321930716],
    [138.5206051676438, 36.30034394260757],
    [138.7465881498856, 36.30034394260757],
    [138.7465881498856, 36.425420321930716],
    [138.5206051676438, 36.425420321930716]  // Closing the polygon
  ]
]);

// Load Landsat 8 Image Collection
var landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
    .filterBounds(roi)
    .filterDate("2023-01-01", "2023-12-31")  // Modify date range as needed
    .sort("CLOUD_COVER")
    .first();

// Select Red, NIR, and Thermal Bands
var red_band = landsat.select("B4");
var nir_band = landsat.select("B5");
var tir_band = landsat.select("B10");  // Thermal Infrared Band

// Compute NDVI
var ndvi = nir_band.subtract(red_band).divide(nir_band.add(red_band)).rename("NDVI");

// Compute Emissivity (Example formula)
var emissivity = ndvi.multiply(0.004).add(0.986).rename("Emissivity");

var K1 = 774.8853;  // Unit: Kelvin
var K2 = 1321.0789; // Unit: Kelvin

// Compute Brightness Temperature (BT) correctly
var bt = tir_band.expression(
    "K2 / log(K1 / TIR + 1)",
    {
        "TIR": tir_band,
        "K1": K1,
        "K2": K2
    }
).rename("Brightness_Temp");

// Compute LST using the corrected formula
var lst = bt.expression(
    "BT / (1 + (10.895 * BT / 14380) * log(E))",
    {
        "BT": bt,
        "E": emissivity
    }
).rename("LST");


// Compute LST min & max dynamically with correct handling
var lstStats = lst.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: roi,
  scale: 30,  // Landsat 8 scale
  maxPixels: 1e13,  
  bestEffort: true
});


// This is server side calculation. minLst and maxLst cannot be computer here.

// ** Use `.evaluate()` to retrieve values **
lstStats.evaluate(function(stats) {
  var lstMin = stats.LST_min;
  var lstMax = stats.LST_max;
  
  print("LST Min:", lstMin);
  print("LST Max:", lstMax);
  
  // Ensure valid min/max values before using them in visualization
  var lstVisParams = {
    min: lstMin - 5,  // Slightly reduce lower bound
    max: lstMax + 5,  // Slightly increase upper bound
    palette: ['blue', 'cyan', 'green', 'yellow', 'red']
  };

  // Add LST Layer to Map (Inside evaluate to ensure correct min/max)
  Map.addLayer(lst.clip(roi), lstVisParams, "LST (Improved)");
});



// Center Map and Add Layers
Map.centerObject(roi, 12);
// Add NDVI Layer (Clipped to ROI)
Map.addLayer(ndvi.clip(roi), 
    {min: 0, max: 1, palette: ['blue', 'white', 'green']}, 
    "NDVI (Clipped)");
    
// Add Emissivity Layer (Clipped to ROI)
Map.addLayer(emissivity.clip(roi), 
    {min: 0.98, max: 1, palette: ['blue', 'white', 'red']}, 
    "Emissivity (Clipped)");


// Print results
print("NDVI", ndvi);
print("Emissivity", emissivity);
print("LST", lst);

