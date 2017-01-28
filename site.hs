--------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings #-}
import           Data.Monoid (mappend)
import           Hakyll
import	qualified Data.Set as S
import			 Text.Pandoc.Options


--------------------------------------------------------------------------------
main :: IO ()
main = hakyll $ do
    match "images/*" $ do
        route   idRoute
        compile copyFileCompiler

    match "css/*" $ do
        route   idRoute
        compile compressCssCompiler


    match "BML/*" $ do
        route $ setExtension "html"
        compile $ pandocMathCompiler
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls
			
    match "Misc/*" $ do
        route $ setExtension "html"
        compile $ pandocMathCompiler
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls

    match "Thesis/*" $ do
        route $ setExtension "html"
        compile $ pandocMathCompiler
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls

    match "OLO/*" $ do
        route $ setExtension "html"
        compile $ pandocMathCompiler
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls
	

--    match "index.html" $ do
--        route idRoute
--        compile $ do
--            posts <- recentFirst =<< loadAll "Misc/*"
--            let indexCtx =
--                    listField "posts" postCtx (return posts) `mappend`
--                    defaultContext
--            getResourceBody
--                >>= applyAsTemplate indexCtx
--                >>= loadAndApplyTemplate "templates/default.html" indexCtx
--                >>= relativizeUrls
--        route idRoute
	

    match "index.md" $ do
        route $ setExtension "html"
        compile $ do 
			posts <- recentFirst =<< loadAll "Misc/*"
			let indexCtx =
				listField "posts" postCtx (return posts) `mappend`
				defaultContext
			pandocMathCompiler >>= applyAsTemplate indexCtx
				>>= loadAndApplyTemplate "templates/files.html" indexCtx
				>>= loadAndApplyTemplate "templates/default.html" indexCtx
				>>= relativizeUrls

    match "olo.md" $ do
        route $ setExtension "html"
        compile $ do 
			posts <- recentFirst =<< loadAll "OLO/*"
			let indexCtx =
				listField "posts" postCtx (return posts) `mappend`
				defaultContext
			pandocMathCompiler >>= applyAsTemplate indexCtx
				>>= loadAndApplyTemplate "templates/files.html" indexCtx
				>>= loadAndApplyTemplate "templates/default.html" indexCtx
				>>= relativizeUrls

    match "bml.md" $ do
        route $ setExtension "html"
        compile $ do 
			posts <- recentFirst =<< loadAll "BML/*"
			let indexCtx =
				listField "posts" postCtx (return posts) `mappend`
				defaultContext
			pandocMathCompiler >>= applyAsTemplate indexCtx
				>>= loadAndApplyTemplate "templates/files.html" indexCtx
				>>= loadAndApplyTemplate "templates/default.html" indexCtx
				>>= relativizeUrls

    match "thesis.md" $ do
        route $ setExtension "html"
        compile $ do 
			posts <- recentFirst =<< loadAll "Thesis/*"
			let indexCtx =
				listField "posts" postCtx (return posts) `mappend`
				defaultContext
			pandocMathCompiler >>= applyAsTemplate indexCtx
				>>= loadAndApplyTemplate "templates/files.html" indexCtx
				>>= loadAndApplyTemplate "templates/default.html" indexCtx
				>>= relativizeUrls


    match "templates/*" $ compile templateBodyCompiler


--------------------------------------------------------------------------------
postCtx :: Context String
postCtx =
    dateField "date" "%B %e, %Y" `mappend`
    defaultContext

pandocMathCompiler :: Compiler (Item String)
pandocMathCompiler =
    let mathExtensions = [Ext_tex_math_dollars, Ext_tex_math_double_backslash,
                          Ext_latex_macros]
        defaultExtensions = writerExtensions defaultHakyllWriterOptions
        newExtensions = foldr S.insert defaultExtensions mathExtensions
        writerOptions = defaultHakyllWriterOptions {
                          writerExtensions = newExtensions,
                          writerHTMLMathMethod = MathJax ""
                        }
    in pandocCompilerWith defaultHakyllReaderOptions writerOptions


