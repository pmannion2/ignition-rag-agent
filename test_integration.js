#!/usr/bin/env node
// Test script for Ignition RAG integration with Cursor

const { enhanceAgentCommand } = require('./cursor_integration');

// Test queries
const testQueries = [
  "How do I use Ignition perspective views?",
  "What are the properties of a tank level tag?",
  "How do I create an Ignition button component?",
  "What JavaScript functions are available in Ignition?",
  "Regular query that shouldn't trigger RAG"
];

// Mock context
const mockContext = {
  currentFile: "example.json",
  language: "json"
};

// Test each query
async function runTests() {
  console.log("Testing Ignition RAG integration with Cursor...\n");
  
  for (const query of testQueries) {
    console.log(`Query: "${query}"`);
    console.log("Enhancing with RAG context...");
    
    try {
      const start = Date.now();
      const enhancedCommand = await enhanceAgentCommand(query, mockContext);
      const duration = Date.now() - start;
      
      const isEnhanced = enhancedCommand !== query;
      console.log(`Enhanced: ${isEnhanced}`);
      console.log(`Time: ${duration}ms`);
      
      if (isEnhanced) {
        // Show preview of enhanced command (first 100 chars)
        const preview = enhancedCommand.substring(0, 100) + (enhancedCommand.length > 100 ? "..." : "");
        console.log(`Preview: ${preview}`);
      }
    } catch (error) {
      console.error(`Error: ${error.message}`);
    }
    
    console.log("-----------------------------------\n");
  }
  
  console.log("Tests completed!");
}

// Run the tests
runTests().catch(console.error); 