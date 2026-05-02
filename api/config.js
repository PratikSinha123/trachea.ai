module.exports = function handler(req, res) {
  res.setHeader("Content-Type", "application/javascript; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");

  const apiBaseUrl = process.env.TRACHEA_API_BASE_URL || "";
  res.status(200).send(
    `window.TRACHEA_API_BASE_URL = ${JSON.stringify(apiBaseUrl)};`
  );
};
