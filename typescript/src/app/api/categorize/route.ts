//Ici, j'ai remplacé le LLM par l'API Python dans Docker. 
//J'ai mis en commentaire l'ancien code.



import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    // 1. On récupère le texte de la réclamation (claimText) envoyé par le front
    const { claimText } = await request.json();

    if (!claimText) {
      return NextResponse.json({ error: 'Le texte de la réclamation est requis.' }, { status: 400 });
    }

    // 2. APPEL API PYTHON DANS DOCKER
    const aiResponse = await fetch('http://127.0.0.1:8000/tags', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_claim: claimText }),
    });

    if (!aiResponse.ok) {
      throw new Error(`L'API Python Docker ne répond pas (Status: ${aiResponse.status})`);
    }

    const aiData = await aiResponse.json();

    // 3. ON RENVOIE LE RÉSULTAT
    return NextResponse.json({ category: aiData.prediction });

  } catch (error) {
    console.error('Erreur lors de la classification locale :', error);
    return NextResponse.json(
      { error: 'Erreur lors de la communication avec l\'API Docker ZenAssist.' },
      { status: 500 }
    );
  }
}








/* import { NextResponse } from 'next/server';
import { Mistral } from '@mistralai/mistralai';

// initialisation Mistral
const apiKey = process.env.MISTRAL_API_KEY;

export async function POST(request: Request) {
  try {
    const { claimText } = await request.json();

    if (!claimText) {
      return NextResponse.json({ error: 'Le texte de la réclamation est requis.' }, { status: 400 });
    }

    if (!apiKey) {
      return NextResponse.json({ error: 'La clé API Mistral n\'est pas configurée.' }, { status: 500 });
    }

    const client = new Mistral({ apiKey });


    const categories_str = "1. Debt collection\n2. Consumer Loan\n3. Credit card or prepaid card\n4. Mortgage\n5. Vehicle loan or lease\n6. Money transfer, virtual currency, or money service\n7. Money transfers\n8. Other (TO BE UPDATED)";

    const prompt = `Tu es un expert en tri de réclamations client chez ZenAssist. 
    Classe la réclamation ci-dessous dans l'UNE de ces catégories : ${categories_str}
    
    CONSIGNE STRICTE : Réponds uniquement par le nom de la catégorie, rien d'autre.
    
    Réclamation : ${claimText}`;

    const chatResponse = await client.chat.complete({
      model: 'mistral-large-latest',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
    });

    // Extraction de la réponse 
    const content = chatResponse.choices?.[0]?.message?.content;
    const expectedCategory = typeof content === 'string' ? content.trim() : '';

    return NextResponse.json({ category: expectedCategory });
  } catch (error) {
    console.error('Erreur API Mistral :', error);
    return NextResponse.json({ error: 'Erreur lors de la communication avec Mistral AI.' }, { status: 500 });
  }
}
*/


