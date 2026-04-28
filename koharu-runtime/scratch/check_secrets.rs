use koharu_runtime::secrets::SecretStore;

fn main() {
    let store = SecretStore::new("koharu");
    match store.get("llm_provider_api_key_gemini") {
        Ok(Some(_)) => println!("Gemini API key found!"),
        Ok(None) => println!("Gemini API key not found."),
        Err(e) => println!("Error accessing secret store: {}", e),
    }
    
    match store.get("llm_provider_api_key_openai") {
        Ok(Some(_)) => println!("OpenAI API key found!"),
        Ok(None) => println!("OpenAI API key not found."),
        Err(e) => println!("Error accessing secret store: {}", e),
    }
}
