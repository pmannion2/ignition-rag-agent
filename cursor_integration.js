/**
 * Cursor Agent Integration for Ignition RAG
 * 
 * This file demonstrates how to integrate the Ignition RAG system with Cursor's agent functionality.
 * It provides JavaScript functions that can be used by Cursor to enhance AI completions with
 * project-specific context.
 */

// Configuration for the RAG API
const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000';

/**
 * Fetches context from the Ignition RAG system for the given query
 * 
 * @param {string} query - The query to search for
 * @param {Object} context - Cursor context information
 * @param {Object} options - Additional options
 * @returns {Promise<string>} The context to add to the prompt
 */
async function getIgnitionContext(query, context = {}, options = {}) {
  const { currentFile, language } = context;
  const { topK = 3, filterType } = options;
  
  try {
    // Determine the filter type if not provided
    let effectiveFilterType = filterType;
    if (!effectiveFilterType && currentFile) {
      if (currentFile.endsWith('.java')) {
        effectiveFilterType = 'tag';
      } else if (currentFile.endsWith('.js') || currentFile.endsWith('.ts')) {
        effectiveFilterType = 'perspective';
      }
    }
    
    // Build the API request
    const response = await fetch(`${RAG_API_URL}/agent/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        top_k: topK,
        filter_type: effectiveFilterType,
        context: currentFile ? { current_file: currentFile } : undefined,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`RAG API request failed with status ${response.status}`);
    }
    
    const data = await response.json();
    
    // Use the suggested prompt if available
    if (data.suggested_prompt) {
      return data.suggested_prompt;
    }
    
    // Otherwise, build our own context format
    const contextChunks = data.context_chunks || [];
    if (contextChunks.length > 0) {
      let contextStr = `Relevant Ignition project context for: ${query}\n\n`;
      
      for (let i = 0; i < contextChunks.length; i++) {
        const chunk = contextChunks[i];
        const source = chunk.source || 'Unknown source';
        const content = chunk.content || '';
        
        contextStr += `--- Context ${i + 1}: ${source} ---\n`;
        contextStr += `${content}\n\n`;
      }
      
      contextStr += `Use the above context to help answer the query or generate appropriate code.\n`;
      return contextStr;
    }
    
    return `No relevant Ignition project context found for: ${query}`;
  } catch (error) {
    console.error('Error fetching Ignition context:', error);
    return `Error fetching Ignition project context: ${error.message}`;
  }
}

/**
 * Intercepts Cursor agent commands to enhance them with Ignition context
 * 
 * @param {string} command - The agent command
 * @param {Object} context - Cursor context information
 * @returns {Promise<string>} The enhanced command with context
 */
async function enhanceAgentCommand(command, context = {}) {
  // Check if the command is related to Ignition
  const ignitionPatterns = [
    /ignition/i,
    /perspective/i,
    /tag(?:s)?/i,
    /view(?:s)?/i,
    /tank level/i,
    /pump status/i,
  ];
  
  const isIgnitionRelated = ignitionPatterns.some(pattern => pattern.test(command));
  
  if (isIgnitionRelated) {
    // Get context from the RAG system
    const ragContext = await getIgnitionContext(command, context);
    
    // Enhance the command with the context
    return `${command}\n\n${ragContext}`;
  }
  
  // Return the original command if not Ignition-related
  return command;
}

// Export the functions for use in Cursor
module.exports = {
  getIgnitionContext,
  enhanceAgentCommand,
};

// Example usage in Cursor agent mode:
/*
const cursor = require('cursor-agent');

cursor.registerCommandEnhancer(async (command, context) => {
  const enhancedCommand = await enhanceAgentCommand(command, {
    currentFile: context.currentFile,
    language: context.language,
  });
  
  return enhancedCommand;
});
*/ 