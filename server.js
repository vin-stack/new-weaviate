const path = require("path");
const api =
  "https://hanna-prodigy-ent-dev-backend-98b5967e61e5.herokuapp.com/chat/";
const fastify = require("fastify")({ logger: true });
const WebSocket = require("ws");
const axios = require("axios");

// Basic route to ensure server is working
fastify.get("/", async (request, reply) => {
  return { message: "Hello, World!" };
});

// Start server
const start = async () => {
  try {
    const address = await fastify.listen({ port: 3000, host: "0.0.0.0" });
    console.log(`Server is running at ${address}`);

    // WebSocket setup
    const wss = new WebSocket.Server({ server: fastify.server });

    wss.on("connection", (ws) => {
      console.log("Client connected");

      ws.on("message", async (message) => {
        try {
          const parsedMessage = JSON.parse(message);
          console.log("Received message:", parsedMessage);

          // Forward the message to the external API
          const apiResponse = await axios.post(api, parsedMessage);

          // Extract the response text from the API response
          const responseText = apiResponse.data;

          // Send the API response text back to the WebSocket client
          ws.send(JSON.stringify({ responseText }));
        } catch (error) {
          console.error("Error processing message:", error);
          ws.send(JSON.stringify({ text: "Error processing message" }));
        }
      });

      ws.on("close", () => {
        console.log("Client disconnected");
      });

      ws.on("error", (error) => {
        console.error("WebSocket error:", error);
      });
    });

    // Periodic ping to clients to keep connections alive
    setInterval(() => {
      wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
          client.ping();
        }
      });
    }, 30000); // Ping every 30 seconds
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();
