// Create UI Elements for User Input
var panel = ui.Panel({
    layout: ui.Panel.Layout.flow('horizontal'),
    style: { position: 'top-left' }
});

// Input fields for Start Date and End Date
var startDateBox = ui.Textbox({
    placeholder: 'Enter Start Date (YYYY-MM-DD)',
    value: '2023-01-01'
});
var endDateBox = ui.Textbox({
    placeholder: 'Enter End Date (YYYY-MM-DD)',
    value: '2023-12-31'
});

// Button to Apply Dates
var applyButton = ui.Button({
    label: 'Fetch Data',
    onClick: function () {
        var startDate = startDateBox.getValue();
        var endDate = endDateBox.getValue();

        // Clear previous layers
        Map.layers().reset();

        // Call function to load and process data
        processLST(startDate, endDate);
    }
});

// Add elements to the panel
panel.add(startDateBox);
panel.add(endDateBox);
panel.add(applyButton);
Map.add(panel);

// Function to Process LST & Related Data
function processLST(startDate, endDate) {
    // Define the ROI using ee.Geometry.Polygon
    var roi = ee.Geometry.Polygon([
        [
            [138.5206051676438, 36.425420321930716],
            [138.5206051676438, 36.30034394260757],
            [138.7465881498856, 36.30034394260757],
            [138.7465881498856, 36.425420321930716],
            [138.5206051676438, 36.425420321930716] // Closing the polygon
        ]
    ]);

    // Load Landsat 8 Image Collection
    var landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
        .filterBounds(roi)
        .filterDate(startDate, endDate)  // Use user-defined dates
        .sort("CLOUD_COVER")
        .first();

    // Select Red, NIR, and Thermal Bands
    var red_band = landsat.select("B4");
    var nir_band = landsat.select("B5");
    var tir_band = landsat.select("B10"); // Thermal Infrared Band

    // Compute NDVI
    var ndvi = nir_band.subtract(red_band).divide(nir_band.add(red_band)).rename("NDVI");

    // Compute Emissivity
    var emissivity = ndvi.multiply(0.004).add(0.986).rename("Emissivity");

    // Planck Constants for Brightness Temperature
    var K1 = 774.8853;
    var K2 = 1321.0789;

    // Compute Brightness Temperature (BT)
    var bt = tir_band.expression(
        "K2 / log(K1 / TIR + 1)",
        {
            "TIR": tir_band,
            "K1": K1,
            "K2": K2
        }
    ).rename("Brightness_Temp");

    // Compute Land Surface Temperature (LST)
    var lst = bt.expression(
        "BT / (1 + (10.895 * BT / 14380) * log(E))",
        {
            "BT": bt,
            "E": emissivity
        }
    ).rename("LST");

    // Compute LST min & max dynamically
    var lstStats = lst.reduceRegion({
        reducer: ee.Reducer.minMax(),
        geometry: roi,
        scale: 30,
        maxPixels: 1e13,
        bestEffort: true
    });

    // Evaluate min/max values
    lstStats.evaluate(function (stats) {
        var lstMin = stats.LST_min;
        var lstMax = stats.LST_max;

        print("LST Min:", lstMin);
        print("LST Max:", lstMax);

        // Define Visualization Parameters
        var lstVisParams = {
            min: lstMin - 5,
            max: lstMax + 5,
            palette: ['blue', 'cyan', 'green', 'yellow', 'red']
        };

        // Add LST Layer to Map
        Map.addLayer(lst.clip(roi), lstVisParams, "LST (Updated)");

        // Add NDVI Layer
        Map.addLayer(ndvi.clip(roi),
            { min: 0, max: 1, palette: ['blue', 'white', 'green'] },
            "NDVI (Updated)");

        // Add Emissivity Layer
        Map.addLayer(emissivity.clip(roi),
            { min: 0.98, max: 1, palette: ['blue', 'white', 'red'] },
            "Emissivity (Updated)");
    });

    // Center Map
    Map.centerObject(roi, 12);
}
