// Cursor Connector for Ignition RAG
// This file connects the cursor_integration.js with Cursor IDE

const fs = require('fs');
const path = require('path');
const { getIgnitionContext, enhanceAgentCommand } = require('./cursor_integration');

// Register with Cursor's agent system
try {
  // Check if we're running in Cursor
  if (typeof cursor !== 'undefined') {
    console.log('Registering Ignition RAG with Cursor Agent...');
    
    // Register the command enhancer
    cursor.registerCommandEnhancer(async (command, context) => {
      const enhancedCommand = await enhanceAgentCommand(command, {
        currentFile: context.currentFile,
        language: context.language,
      });
      
      return enhancedCommand;
    });
    
    console.log('Successfully registered Ignition RAG with Cursor Agent.');
  } else {
    console.log('Not running in Cursor environment, skipping registration.');
  }
} catch (error) {
  console.error('Error registering with Cursor Agent:', error.message);
}

// Export for use in other contexts
module.exports = {
  getIgnitionContext,
  enhanceAgentCommand,
}; 