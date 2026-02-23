import axios from "axios";

export async function getWeather(apiKey, location) {
  const url =
    "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/" +
    encodeURIComponent(location) +
    "?unitGroup=metric&include=current&contentType=json&key=" +
    apiKey;

  const res = await axios.get(url);

  return res.data.currentConditions;
}